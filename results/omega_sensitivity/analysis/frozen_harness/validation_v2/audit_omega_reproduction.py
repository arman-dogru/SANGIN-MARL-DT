"""Exact execution reproduction with solver-native certificates for fractional diagnostics."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import gzip
import json
from pathlib import Path
import pickle
import time

from panama_uav.records import sha256


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def certificate(before,after):
    """Do not require identical minimizers from a finite-tolerance conic solve."""
    def tolerance(result):
        attempts=result.get('solver_attempts',[])
        return float(attempts[-1]['tolerance']) if attempts else 1e-8
    allowed=2*max(tolerance(before),tolerance(after))*max(1.,abs(before['objective']),abs(after['objective']))
    difference=abs(before['objective']-after['objective'])
    residual=max(before['residual'],after['residual'])
    passed=(before['status']==after['status']=='optimal' and before['is_dcp'] and after['is_dcp'] and residual<=1e-6 and difference<=allowed)
    return dict(passed=bool(passed),objective_difference=float(difference),objective_tolerance=float(allowed),max_residual=float(residual))


def reference_signatures(argument):
    import run_omega_sensitivity as driver
    root,weight,condition,case=argument;root=Path(root);m=read(root/'manifest.json')
    reference=m['baseline_references'][condition['name']+'/'+case['case_id']]
    if sha256(reference['packet'])!=reference['packet_sha256'] or sha256(reference['archive'])!=reference['archive_sha256']:
        raise ValueError('Historical baseline changed')
    old=read(reference['packet']);current=read(root/'episodes/W1'/condition['name']/(case['case_id']+'.result.json'))
    with gzip.open(reference['archive'],'rt',encoding='utf-8') as stream:oldtrace=json.load(stream)
    expected=driver.signatures(old['row'],old['tasks'],old['rounds'],oldtrace)
    before={e['new_job']:e['relaxed'] for e in oldtrace['transport_events'] if e['event']=='paper_allocation'}
    del oldtrace
    if sha256(current['audit']['archive'])!=current['audit']['archive_sha256']:raise ValueError('New archive changed')
    with gzip.open(current['audit']['archive'],'rb') as stream:newtrace=pickle.load(stream)
    after={e['new_job']:e['relaxed'] for e in newtrace['transport_events'] if e['event']=='paper_allocation'}
    del newtrace
    if before.keys()!=after.keys():raise ValueError('Fractional snapshot IDs changed')
    certificates=[dict(job_id=k,**certificate(before[k],after[k])) for k in before]
    matches={name:expected[name]==current['signatures'][name] for name in expected}
    relaxed=max((abs(float(a.get('relaxed_x',0))-float(b.get('relaxed_x',0))) for a,b in zip(old['tasks'],current['tasks'])),default=0.)
    return dict(protocol='execution_exact_fractional_certificate_v2',condition=condition['name'],case_id=case['case_id'],num_agents=case['num_agents'],
        matches=matches,relaxed_x_max_absolute_difference=relaxed,fractional_certificates=len(certificates),
        max_fractional_objective_difference=max((c['objective_difference'] for c in certificates),default=0.),
        max_fractional_residual=max((c['max_residual'] for c in certificates),default=0.),
        failed_certificates=[c for c in certificates if not c['passed']],
        passed=all(matches.values()) and all(c['passed'] for c in certificates),
        historical_packet_sha256=reference['packet_sha256'],historical_archive_sha256=reference['archive_sha256'])


def baseline_gate(root,jobs,output='baseline_validation.json'):
    import run_omega_sensitivity as driver
    root=Path(root);folder=root/'baseline_audit_v2';folder.mkdir(exist_ok=True);results=[]
    with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1) as executor:
        for weight,condition,case in jobs:
            if time.time()>read(root/'launch.json')['first_started_unix']+read(root/'manifest.json')['budget_seconds']:
                raise TimeoutError('Recorded study wall cap reached during reproduction audit')
            path=folder/(condition['name']+'__'+case['case_id']+'.json')
            if path.exists():result=read(path)
            else:
                result=executor.submit(reference_signatures,(str(root),weight,condition,case)).result();driver.write(path,result)
            results.append(result)
            if not result['passed']:
                driver.write(root/output,dict(status='failed',protocol='fractional_certificate_v2',cases=results))
                raise ValueError('Baseline reproduction failed: '+str(path))
    driver.write(root/output,dict(status='passed',protocol='execution_exact_fractional_certificate_v2',cases=len(results),
        all_virtual_time_records_match=True,host_measurements_excluded=sorted(driver.HOST_FIELDS),
        fractional_diagnostic_rule='Both solves optimal/DCP, residual <= 1e-6, objective difference <= twice the recorded conic tolerance times max(1,objective magnitude); executed records exact',
        details=results))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--completed-up-to',type=int,default=16);args=parser.parse_args()
    root=args.root.resolve();m=read(root/'manifest.json');cases=read(root/'cases.json')
    jobs=[(m['weights'][0],c,case) for c in m['conditions'] for case in cases if case['num_agents']<=args.completed_up_to
        and (root/'episodes/W1'/c['name']/(case['case_id']+'.result.json')).exists()]
    baseline_gate(root,jobs,output='baseline_validation_candidate.json')
    print(json.dumps(dict(status='passed',completed_case_certificates=len(jobs))))
