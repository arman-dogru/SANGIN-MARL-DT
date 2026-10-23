"""Explain aggregate age and non-monotone admission effects using saved traces."""
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve()
conditions=('allocated__nominal','allocated__uplink-0.05','cloud__nominal',
            'cloud__ideal_network','cloud__aoi-3')
results={}
for name in conditions:
    counters=Counter();availability=defaultdict(Counter);age=defaultdict(Counter)
    for path in sorted((root/'episodes'/name).glob('agents-8-*.json.gz')):
        with gzip.open(path,'rt',encoding='utf-8') as stream:trace=json.load(stream)
        counters['cases']+=1
        for task in trace['resource_tasks']:
            radio=task.get('backhaul_radio',{})
            state='available' if radio.get('available',True) else 'unavailable'
            availability[state]['tasks']+=1
            availability[state]['uav']+=task.get('executed_x')==1
            availability[state]['cloud']+=task.get('executed_x')==0
            availability[state][task['deadline_status']]+=1
        for event in trace['oracle_events']:
            if event['event']!='controlled_peer_shadow':continue
            for pair in event['pairs']:
                peer=event['fresh_states'][str(pair['peer'])]
                category='completed_peers' if peer['completed'] else 'active_peers'
                age[category]['pairs']+=1
                age[category]['age_seconds_sum']+=pair['peer_age_seconds']
                age[category]['relevant_pairs']+=pair['physical_relevant']
        del trace
    for group in age.values():group['mean_age_seconds']=group['age_seconds_sum']/group['pairs'] if group['pairs'] else None
    results[name]=dict(cases=counters['cases'],admission_radio={k:dict(v) for k,v in availability.items()},
                       peer_age={k:dict(v) for k,v in age.items()})
result=dict(manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
    population=8,source='all ten cases in each listed condition; archived tasks and controlled_peer_shadow records',
    conditions=results)
(root/'mechanism_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
