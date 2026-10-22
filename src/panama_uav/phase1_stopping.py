"""Predeclared bounded validation stopping; never use held-out outcomes."""
import math
from .phase1_data import FAMILIES

COUNTS=(2,4,6,8)


def covered(episodes, minimum=500):
    return minimum>=500 and all(int(episodes.get(n,episodes.get(str(n),0)))>=minimum for n in COUNTS)


def training_stop(history, episodes, *, patience=6, min_delta=.01, min_steps=90000):
    if patience<2 or min_delta<=0:
        raise ValueError('A finite positive stagnation criterion is required')
    valid=[]
    for row in history:
        groups=row.get('populations',[])
        if [r['num_agents'] for r in groups]!=list(COUNTS) or not all(r.get('complete') for r in groups):
            continue
        rates=[r['success_rate'] for r in groups]
        if any(v is None or not math.isfinite(v) or not 0<=v<=1 for v in rates):
            raise ValueError('Invalid validation rate')
        valid.append((row,sum(rates)/len(rates)))
    if not valid or not covered(episodes):
        return None
    latest,score=valid[-1]
    # An early freeze remains subject to independent full 200-case cohorts;
    # this small validation sample cannot itself confer qualification.
    ready=all(r['success_rate']>=.95 and set(r.get('per_family',{}))==set(FAMILIES)
        and all(g.get('success_rate',0)>=.95 for g in r['per_family'].values())
        for r in latest['populations'])
    if ready:
        return 'validation_ready'
    if latest['generated_joint_steps']<min_steps or len(valid)<=patience:
        return None
    old_best=max(score for _,score in valid[:-patience])
    recent=[score for _,score in valid[-patience:]]
    recent_best=max(recent)
    recovering=max(recent[patience//2:])>max(recent[:patience//2])+min_delta
    if recent_best<=old_best+min_delta and not recovering:
        return 'validation_plateau'
    return None


def may_open_heldout(validation, episodes, compatibility):
    return (covered(episodes) and len(validation)==4 and
        [r['num_agents'] for r in validation]==list(COUNTS) and
        all(r.get('episodes')==200 and r.get('complete') and r.get('passed') and r.get('success_rate',0)>=.95
            and r.get('exposure_passed') and r.get('relevant_pairs',0)>=200
            and set(r.get('per_family',{}))==set(FAMILIES)
            and all(g.get('episodes')==40 and g.get('success_rate',0)>=.95 and g.get('exposed_cases',0)>=20
                for g in r['per_family'].values()) for r in validation) and
        compatibility is not None and compatibility.get('status')=='passed' and
        compatibility.get('execution_contract')=='remote-aligned-v2')
