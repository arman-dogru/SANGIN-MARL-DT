"""Local-only training curves, final outcomes and unqualified failure reporting."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def table(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);status=json.loads((root/'status.json').read_text())
    rows=[json.loads(s) for s in (root/'training.jsonl').read_text().splitlines()] if (root/'training.jsonl').exists() else []
    table(root/'training.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    if rows:
        window=min(100,len(rows));x=np.array([r['generated_joint_steps'] for r in rows])
        for ax,field,label in ((axes[0],'success','Rolling training success (exploratory)'),(axes[1],'steps','Rolling episode steps')):
            values=np.convolve([float(r[field]) for r in rows],np.ones(window)/window,mode='valid')
            ax.plot(x[window-1:],values);ax.set(xlabel='Generated joint steps',ylabel=label);ax.grid(alpha=.2)
    fig.suptitle('Phase 1: clean MAPF training; qualification uses greedy held-out cohorts')
    fig.savefig(root/'training_curves.png',dpi=150,bbox_inches='tight');plt.close(fig)
    flat=[];failures=[]
    for split,key in (('validation','qualification'),('test','held_out')):
        for item in status.get(key,[]):
            for family,group in [('all',item),*item.get('per_family',{}).items()]:
                flat.append(dict(split=split,num_agents=item['num_agents'],family=family,
                    **{k:v for k,v in group.items() if k not in ('per_family','num_agents')}))
        path=root/f'final_{split}_cases.jsonl'
        if path.exists():
            cases=[json.loads(s) for s in path.read_text().splitlines()]
            table(root/f'final_{split}_cases.csv',cases)
            failures.extend(dict(r,split=split) for r in cases if not r['success'])
    table(root/'per_population_and_scenario.csv',flat);table(root/'failures.csv',failures)
    lines=['# Phase 1 clean PANAMA MAPF training','',
        f"Status: **{status['status']}**. Qualified: **{status['qualified']}**. Runtime: {status['wall_seconds']/3600:.3f} h "
        f"of {status['budget_seconds']/3600:.1f} h. Episodes: {status['episodes']}; joint steps: {status['generated_joint_steps']}; "
        f"learner updates: {status['learner_updates']}.",'',
        'No communication delays, allocation or SAGIN service are used in training. The frozen final policy is evaluated on '
        'disjoint validation and test maps after learning stops. Test results are never used to choose weights or resume learning. '
        'Curriculum qualification history is not substituted for final-current-model performance.','',
        '| Split | Robots | Completed / 200 | Success | 95% Wilson CI | Exposure pairs | Passed |',
        '| --- | ---: | --- | --- | --- | ---: | --- |']
    for split,key in (('validation','qualification'),('test','held_out')):
        for item in status.get(key,[]):
            rate='missing' if item['success_rate'] is None else f"{100*item['success_rate']:.1f}%"
            lo,hi=item['success_ci95'];interval='missing' if lo is None else f'[{100*lo:.1f}, {100*hi:.1f}]%'
            lines.append(f"| {split} | {item['num_agents']} | {item['episodes']} / 200 | {rate} | {interval} | {item['relevant_pairs']} | {item['passed']} |")
    lines+=['','Qualification requires at least 95% success in every 200-case population and every 40-case scenario family, '
        'sufficient coordination exposure, all four curriculum stages completed, and passing both final cohorts. Smoke runs '
        'can never qualify. Partial evaluations remain explicitly partial. Cases sharing a map are correlated: Wilson intervals '
        'are nominal case-level summaries; map-cluster intervals are retained where available.','',
        'The canonical observation uses the unchanged remote-DT constructor with fresh states. This explicitly aligns the '
        'waypoint and square-FoV conventions; the historical local guidance-consumption offset is not reproduced. The original '
        'local physical resolver/rewards remain unchanged. Remote post-completion local control remains a separate existing '
        'behavior, so active-observation compatibility does not claim complete mission equivalence.','',
        'See manifest.json, data/dataset.json, status.json, per_population_and_scenario.csv, failures.csv, '
        'compatibility.json and training_curves.png. A failed or incomplete run is not a qualified baseline.','']
    (root/'PHASE1_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return status
