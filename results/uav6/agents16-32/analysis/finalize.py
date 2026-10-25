"""Final record accounting and index for the local population extension."""
from collections import Counter, defaultdict
import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import numpy as np
from panama_uav.records import sha256, write_json
from panama_uav.communication_report import csv_table, paired_changes

parser = argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('output/uav6_pilot/uav6-20261024-01/extensions/agents16-32-01'))
root = parser.parse_args().root.resolve()
m = json.loads((root/'manifest.json').read_text())
base = Path(m['base_experiment'])
assert json.loads((root/'status.json').read_text())['status'] == 'complete'
rows, expected_tasks, expected_rounds, archives = [], {}, {}, []
for path in sorted((base/'episodes').glob('*/*.result.json')):
    rows.append(json.loads(path.read_text())['row'])
for path in sorted((root/'episodes').glob('*/*.result.json')):
    packet = json.loads(path.read_text())
    rows.append(packet['row'])
    assert packet['manifest_sha256'] == sha256(root/'manifest.json')
    audit = packet['audit']
    assert audit['status'] == 'passed' and sha256(audit['archive']) == audit['archive_sha256']
    archives.append(audit)
    for name, index, identity in (('tasks', expected_tasks, ('condition','case_id','job_id')),
                                   ('rounds', expected_rounds, ('condition','case_id','cycle'))):
        for value in packet[name]:
            key = tuple(value[k] for k in identity)
            assert key not in index
            index[key] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
assert len(rows) == len({(r['condition'],r['case_id']) for r in rows}) == 100
counts, statuses, cutoff = {}, Counter(), 0
for name, expected, identity in (('tasks', expected_tasks, ('condition','case_id','job_id')),
                                 ('rounds', expected_rounds, ('condition','case_id','cycle'))):
    observed = set()
    with (root/(name+'.jsonl')).open() as stream:
        for line in stream:
            value = json.loads(line)
            key = tuple(value[k] for k in identity)
            assert key not in observed and key in expected
            assert hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() == expected[key]
            observed.add(key)
            if name == 'tasks':
                statuses[value['deadline_status']] += 1
                cutoff += value['cutoff_induced_wait']
    assert observed == set(expected)
    counts[name] = len(observed)
groups = defaultdict(list)
for row in rows:
    groups[row['condition'],row['num_agents']].append(row)
assert all(len(v) == 5 for v in groups.values()) and len(groups) == 20
pairwise, coordination, placement = [], [], []
names = [r['name'] for r in m['conditions']]
for population in (2,4,8,16,32):
    for reference, actual in itertools.combinations(names, 2):
        for metric in ('mission_success_rate','command_on_time_rate','case_command_p95_seconds',
                       'makespan_seconds','waiting_steps','modeled_energy_j'):
            pairwise.append(dict(condition=actual,reference=reference,num_agents=population,
                **paired_changes(groups[actual,population],groups[reference,population],metric)))
    for condition in names:
        group = groups[condition,population]
        point = dict(condition=condition,num_agents=population,cases=len(group))
        for label, numerator, denominator in (
            ('cached_fov','fov_errors','fov_relevant_pairs'),
            ('inference_physical_fov','physical_fov_order_errors','physical_relevant_pairs'),
            ('stale_fresh_actions','stale_fresh_disagreements','inferences'),
            ('cutoff_oracle_actions','decision_action_shadow_disagreements','decision_action_shadow_comparisons')):
            a, b = sum(r[numerator] for r in group), sum(r[denominator] for r in group)
            point.update({label+'_numerator':a,label+'_denominator':b,label+'_rate':a/b if b else None})
        coordination.append(point)
        cloud = np.array([r['executed_cloud_tasks'] for r in group])
        total = np.array([r['executed_cloud_tasks']+r['executed_drone_tasks'] for r in group])
        indices = np.random.default_rng(20261024).integers(0,len(group),(2000,len(group)))
        low, high = np.quantile(cloud[indices].sum(axis=1)/total[indices].sum(axis=1),[.025,.975])
        placement.append(dict(condition=condition,num_agents=population,cases=5,
            cloud_fraction=float(cloud.sum()/total.sum()),ci95_low=float(low),ci95_high=float(high),bootstrap_seed=20261024))
csv_table(root/'combined_pairwise.csv',pairwise)
csv_table(root/'combined_coordination.csv',coordination)
csv_table(root/'placement_intervals.csv',placement)
bm = json.loads((base/'manifest.json').read_text())
versions = {name:importlib.metadata.version(name) for name in bm['packages']}
assert versions == bm['packages']
write_json(root/'environment.json',dict(packages=versions,base_versions_match=True))
for name, digest in m['protected_sha256'].items():
    assert sha256(Path(m['project_root'])/name) == digest
reference = json.loads((root/'mapf_reference_serial.json').read_text())
assert reference['inference_comparisons'] == 6682
assert reference['batch_serial_disagreements'] == 0
assert reference['case_bank_sha256'] == sha256(root/'cases.json')
assert reference['model_sha256'] == m['model_sha256']
case_outcomes = {}
for row in rows:
    point = case_outcomes.setdefault(row['case_id'],dict(case_id=row['case_id'],
        num_agents=row['num_agents'],seed=row['seed']))
    point[row['condition']+'_success'] = row['success']
    point[row['condition']+'_makespan'] = row['makespan_seconds']
for row in reference['records']:
    point = case_outcomes[row['case_id']]
    point['instantaneous_mapf_success'] = row['success']
    point['instantaneous_mapf_steps'] = row['steps']
csv_table(root/'case_outcomes.csv',list(case_outcomes.values()))
summary = json.loads((root/'combined_summary.json').read_text())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42,
                     'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), layout='constrained')
fields = (('cloud_fraction','Cloud tasks over mission (%)',100),
          ('observed_p95_s','Observed E2E command p95 (ms)',1000),
          ('on_time_fraction','Commands within 250 ms (%)',100),
          ('drone_utilization','UAV CPU utilization (%)',100),
          ('cloud_utilization','Cloud CPU utilization (%)',100),
          ('completed_missions','Mission success (%)',100))
for condition, color, marker, label in zip(names,
        ('#666666','#009E73','#0072B2','#D55E00'),('o','s','^','D'),
        ('Allocator 1 GHz','Allocator 6 GHz','All UAV 6 GHz','All cloud')):
    group = sorted([r for r in summary if r['condition']==condition],key=lambda r:r['num_agents'])
    for ax, (field, ylabel, scale) in zip(axes.flat, fields):
        values = [scale*r[field]/r['cases'] if field=='completed_missions' else scale*r[field] for r in group]
        ax.plot(range(len(group)),values,color=color,marker=marker,label=label,lw=1.4,ms=4)
for ax, (field, ylabel, scale) in zip(axes.flat, fields):
    ax.set(xlabel='Robots',ylabel=ylabel,xticks=range(5),xticklabels=('2','4','8','16','32'))
    ax.grid(alpha=.2)
    if field != 'observed_p95_s':
        ax.set_ylim(-3,103)
axes[0,1].axhline(250,color='black',ls=':',lw=1)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncols=4,frameon=False)
fig.suptitle('Parameter-only pilot results - extended experiment set',fontsize=12)
for extension in ('pdf','svg','png'):
    fig.savefig(root/'figures'/('combined_diagnostics.'+extension),dpi=300,
                metadata={'CreationDate':None} if extension=='pdf' else None)
plt.close(fig)
initial = sorted([r for r in summary if r['condition']=='allocated-6ghz'],key=lambda r:r['num_agents'])
fig, ax = plt.subplots(figsize=(7.2,4.3),layout='constrained')
positions = np.arange(len(initial))
drone = [r['first_round_uav_mean'] for r in initial]
cloud_counts = [r['first_round_cloud_mean'] for r in initial]
drone_percent = [100*d/(d+c) for d,c in zip(drone,cloud_counts)]
cloud_percent = [100*c/(d+c) for d,c in zip(drone,cloud_counts)]
lower = ax.bar(positions,drone_percent,color='#0072B2',label='UAV')
upper = ax.bar(positions,cloud_percent,bottom=drone_percent,color='#D55E00',label='Cloud')
ax.bar_label(lower,labels=[f'{v:.1f} tasks' for v in drone],label_type='center',color='white',fontsize=8)
ax.bar_label(upper,labels=[f'{v:.1f} tasks' for v in cloud_counts],label_type='center',color='white',fontsize=8)
ax.set(xlabel='Robots',ylabel='First-round placement (%)',
       xticks=positions,xticklabels=[r['num_agents'] for r in initial],ylim=(0,103))
ax.set_axisbelow(True)
ax.yaxis.grid(alpha=.2)
fig.legend(*ax.get_legend_handles_labels(),frameon=False,loc='outside lower center',ncols=2)
ax.set_title('Parameter-only pilot: first-round placement, allocator 6 GHz')
for extension in ('pdf','svg','png'):
    fig.savefig(root/'figures'/('first_round_placement.'+extension),dpi=300,
                metadata={'CreationDate':None} if extension=='pdf' else None)
plt.close(fig)
report = (root/'REPORT.md').read_text(encoding='utf-8').split('\n## Combined analysis')[0]
report = report.replace('\u00e2\u20ac\u201d', ' - ')
report += '\n## Combined analysis\n\n'
report += 'The full set contains 100 primary simulations: five paired cases at each of 2/4/8/16/32 robots, with four conditions. The instantaneous MAPF check is a separate reference on the ten added cases.\n\n'
report += '| Robots | Instantaneous MAPF success |\n| --- | --- |\n'
for value in reference['summary']:
    report += f"| {value['num_agents']} | {value['successes']}/{value['cases']} |\n"
report += '\nCommand timing and delayed state availability can change joint trajectories in both directions with a frozen policy. The reference and networked mission outcomes are reported separately.\n\n'
report += 'The matching instantaneous reference uses per-agent inference; 6,682 comparisons with batched inference produced identical actions.\n\n'
report += '6 GHz allocator cloud shares: '+', '.join(
    f"{r['num_agents']} robots {100*r['cloud_fraction']:.2f}%"
    for r in sorted(summary,key=lambda x:x['num_agents']) if r['condition']=='allocated-6ghz')+'.\n\n'
report += '| Robots | First-round UAV tasks (mean) | First-round cloud tasks (mean) | Cloud tasks over mission (%) |\n| --- | --- | --- | --- |\n'
for row in sorted(summary,key=lambda r:r['num_agents']):
    if row['condition']=='allocated-6ghz':
        report += f"| {row['num_agents']} | {row['first_round_uav_mean']:.2f} | {row['first_round_cloud_mean']:.2f} | {100*row['cloud_fraction']:.2f} |\n"
report += '\n'
report += ('Initial-round allocations favor UAV execution at 2/4/8 robots and cloud execution at 16/32 robots. '
    'Mission-wide percentages also include later rounds with fewer active robots. '
    '`figures/first_round_placement.png` shows the initial task counts; `placement_by_active_count.csv` retains the active-population strata.\n\n')
report += '`combined_pairwise.csv` provides every paired treatment contrast with 95% intervals. Summary latency quantiles pool delivered tasks; paired p95 contrasts compare per-case p95 values. `placement_intervals.csv` gives case-clustered placement intervals. `combined_coordination.csv` retains ordering/action-disagreement numerators and denominators. `accounting_audit.json` verifies record identity and preservation. These are small parameter-only pilot cohorts on one map.\n'
report += '\n`case_outcomes.csv` lists paired mission outcomes, including the supplementary instantaneous reference for the added populations.\n'
report += '\n### Findings at 16 and 32 robots\n\n'
lookup = {(r['condition'],r['num_agents']):r for r in summary}
for population in (16,32):
    old, new = (lookup[c,population] for c in ('allocated-1ghz','allocated-6ghz'))
    report += (f"At {population} robots, the 1 GHz and 6 GHz allocators completed "
        f"{old['completed_missions']}/{old['cases']} and {new['completed_missions']}/{new['cases']} missions. "
        f"On-time delivery changed from {100*old['on_time_fraction']:.2f}% to {100*new['on_time_fraction']:.2f}%; "
        f"observed command p95 changed from {1000*old['observed_p95_s']:.2f} to {1000*new['observed_p95_s']:.2f} ms. "
        f"Modeled energy per episode changed from {old['modeled_energy_mean_episode_j']:.3f} "
        f"to {new['modeled_energy_mean_episode_j']:.3f} J.\n\n")
cloud = lookup['cloud-6ghz',32]
report += (f"All-cloud execution at 32 robots completed {cloud['completed_missions']}/{cloud['cases']} missions, "
    f"with {100*cloud['on_time_fraction']:.2f}% on-time commands. "
    'The paired outcomes include mixed changes in mission completion; command timeliness alone does not explain every outcome.\n\n')
uav = lookup['uav-6ghz',32]
report += (f"All-UAV execution at 32 robots completed {uav['completed_missions']}/{uav['cases']} missions, "
    f"with {100*uav['on_time_fraction']:.2f}% on-time commands, "
    f"{100*uav['drone_utilization']:.2f}% UAV CPU utilization and "
    f"{uav['drone_concurrent_max']} concurrent jobs at peak. "
    'Round-level contention and the 250 ms cutoff remain important even with unused capacity between control rounds.\n\n')
shares = [lookup['allocated-6ghz',n]['cloud_fraction'] for n in (2,4,8,16,32)]
if any(b<a for a,b in zip(shares,shares[1:])):
    report += 'Cloud placement is not monotonic across the full population range. '
report += ('The 6 GHz setting improves command timeliness in the added cohorts. '
    'The paired mission counts and first-round versus mission-wide placement table show the limits of interpreting capacity changes as a fixed placement transition.\n')
(root/'REPORT.md').write_text(report,encoding='utf-8')
index = '''# UAV CPU experiment set

The same four execution conditions cover 2, 4, 8, 16 and 32 robots, with five
paired cases per population and a 250-step cap. The full set has 100 primary
simulations. Existing records are retained unchanged; the added populations
are stored as an extension.

- [Combined report](extensions/agents16-32-01/REPORT.md)
- [Combined summary](extensions/agents16-32-01/combined_summary.csv)
- [Paired comparisons](extensions/agents16-32-01/combined_pairwise.csv)
- [Paired mission outcomes](extensions/agents16-32-01/case_outcomes.csv)
- [Combined diagnostic plot](extensions/agents16-32-01/figures/combined_diagnostics.png)
- [First-round placement plot](extensions/agents16-32-01/figures/first_round_placement.png)
- [Instantaneous MAPF reference](extensions/agents16-32-01/mapf_reference_serial.json)
- [Record/accounting audit](extensions/agents16-32-01/accounting_audit.json)

Parameters remain 6 GHz UAV / 10 GHz cloud (1 GHz UAV for the original allocator),
100000 bits at 1000 cycles/bit, 10/50 Mbps LEO, kappa 1e-28, objective coefficients
1/1, a one-second control interval and 250 ms cutoffs. No learning updates occur.
All artifacts remain local.
'''
target = base/'EXPERIMENT_SET.md'
if target.exists() and target.read_text(encoding='utf-8') != index:
    owner = json.loads((root/'set_index.json').read_text())
    assert Path(owner['extension']) == root and Path(owner['base_experiment']) == base
target.write_text(index,encoding='utf-8')
audit = dict(status='passed',primary_episodes=100,new_episodes=40,new_replay_audits=len(archives),
    new_aggregate_counts=counts,new_command_partition={s:statuses[s] for s in ('on_time','late','undelivered')},
    new_cutoff_waits=cutoff,atomic_transactions_match=True,paired_case_identity=True,
    preserved_source_and_history_files=len(m['protected_sha256']),package_versions_unchanged=True,policy_updates=0,
    model_sha256=m['model_sha256'],timeline_sha256=m['timeline_sha256'],
    extension_manifest_sha256=sha256(root/'manifest.json'),postprocessor_sha256=sha256(__file__),
    supplementary_reference_action_comparisons=reference['inference_comparisons'],
    supplementary_reference_action_disagreements=reference['batch_serial_disagreements'])
write_json(root/'accounting_audit.json',audit)
write_json(root/'set_index.json',dict(populations=[2,4,8,16,32],cases_per_population=5,conditions=names,
    primary_episodes=100,base_experiment=str(base),extension=str(root),report='REPORT.md',supplementary_mapf_cases=10,
    base_manifest_sha256=m['base_manifest_sha256'],extension_manifest_sha256=sha256(root/'manifest.json')))
print(json.dumps(audit))
