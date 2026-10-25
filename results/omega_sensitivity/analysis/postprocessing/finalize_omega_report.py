"""Format reviewed numerical findings and uncertainty from completed study tables."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from panama_uav.communication_report import csv_table
from panama_uav.records import sha256

COUNTS=(2,4,8,16,32)
WEIGHTS=('W1','W2','W3','W4','W5')


def rows(path):return list(csv.DictReader(Path(path).open(encoding='utf-8')))


def finalize(root):
    root=Path(root).resolve();summary=rows(root/'summary.csv');table={(r['weight_id'],r['route'],int(r['num_agents'])):r for r in summary}
    comparisons=rows(root/'comparisons.csv');effects={}
    for r in comparisons:
        if r['route']==r['reference_route']=='allocated' and r['reference_weight']=='W1':
            effects[(r['weight_id'],int(r['num_agents']),r['metric'])]=r
    folder=root/'analysis';dominance=[]
    for n in COUNTS:
        for w in WEIGHTS:
            a=table[w,'allocated',n];weak=strict=True
            for route in ('edge','cloud'):
                b=table[w,route,n]
                latency=float(a['observed_latency_p95_seconds'])-float(b['observed_latency_p95_seconds'])
                energy=float(a['mean_episode_energy_j'])-float(b['mean_episode_energy_j'])
                weak&=latency<=1e-9 and energy<=1e-9 and (latency < -1e-9 or energy < -1e-9)
                strict&=latency < -1e-9 and energy < -1e-9
            dominance.append(dict(weight_id=w,num_agents=n,strict_latency_and_energy_dominance=strict,
                no_worse_latency_and_energy_with_one_strict_improvement=weak,tolerance=1e-9))
    csv_table(folder/'control_dominance.csv',dominance)
    energy_gains=[100*(1-float(table['W2','allocated',n]['mean_episode_energy_j'])/float(table['W1','allocated',n]['mean_episode_energy_j'])) for n in COUNTS]
    lines=['## Findings','',
        'The tested weights change CPU rates, task identities and placement, but the cohort shows a modest transition in mission-wide routing. '
        'All settings use mixed UAV/cloud execution with two robots. None becomes cloud-majority at 16 robots. '
        'At 32 robots, cloud shares range from 48.83% to 57.79%; W2 has the largest cloud share. The hypothesized sharp low-population UAV to high-population cloud transition is not established by these cases.','',
        f"W2 (latency 1, energy 2) provides a descriptive compromise: mean episode energy is {min(energy_gains):.1f}–{max(energy_gains):.1f}% below W1, "
        'with the same success counts at every population. Smaller-population latency increases remain within the 250 ms cutoff. '
        'At 32 robots, the energy and per-case p95 confidence intervals include zero; the two successful missions also have different case identities, with only one common completed pair. '
        'The paired makespan result there rests on that single pair. These observations support follow-up testing of W2, rather than a general performance guarantee.','',
        'W3 gives the lowest mean modeled energy at every population, with higher observed p95 than W1. Its 32-robot success count is 1/5 versus W1’s 2/5. '
        'Higher latency coefficients give uneven tail-latency changes: W4 increases four-robot p95, and W5 increases eight- and 32-robot p95. '
        'W4 has the lowest allocated miss rate at 16 and 32 robots, while success falls from 5/5 to 4/5 and from 2/5 to 1/5, respectively. '
        'W5 reaches 3/5 successes at 32 robots while its miss rate rises to 5.40%. Deadline compliance and mission completion therefore do not improve together consistently in this cohort.','',
        '### Paired W2 minus W1 effects','',
        'Energy uses all five paired episodes. Latency is the mean difference in per-case p95, which differs from subtracting the pooled p95 values in the comparison table. Intervals are paired case-bootstrap 95% intervals.','',
        '| Robots | Energy difference (J/episode), 95% CI | Per-case p95 difference (ms), 95% CI | W1 / W2 successes |',
        '|---:|---:|---:|:---:|']
    excerpt=[]
    for n in COUNTS:
        e=effects['W2',n,'modeled_energy_j'];l=effects['W2',n,'case_command_p95_seconds']
        lines.append(f"| {n} | {float(e['difference']):+.3f} [{float(e['ci95_low']):+.3f}, {float(e['ci95_high']):+.3f}] | "
            f"{float(l['difference'])*1000:+.2f} [{float(l['ci95_low'])*1000:+.2f}, {float(l['ci95_high'])*1000:+.2f}] | "
            f"{table['W1','allocated',n]['successful_missions']}/5 / {table['W2','allocated',n]['successful_missions']}/5 |")
        excerpt.extend((e,l,effects['W2',n,'mission_success_rate']))
    csv_table(folder/'w2_paired_effects.csv',excerpt)
    report=(root/'report.md').read_text(encoding='utf-8')
    if '\n## Findings\n' in report:
        a=report.index('\n## Findings\n');b=report.index('\n## Comparative results\n',a);report=report[:a]+report[b:]
    report=report.replace('\n## Comparative results\n','\n'+'\n'.join(lines)+'\n\n## Comparative results\n',1)
    a=report.index('## Comparative results');b=report.index('## Research questions',a)
    original=report[a:b]
    if '| Robots |' in original:
        (folder/'comparison_tables.md').write_text('# Complete 75-cell comparison\n\n'+original.split('\n',1)[1],encoding='utf-8')
    report=(report[:a]+'## Comparative results\n\nThe full 75-cell table is in [comparison_tables.md](analysis/comparison_tables.md). '
        '[summary.csv](summary.csv) includes all metrics; [episodes.csv](episodes.csv) retains all 375 case results.\n\n'+report[b:])
    a=report.find('\n- **2 robots:**');b=report.find('\n1. **Higher latency/energy ratio:**',a)
    if a>=0 and b>a:report=report[:a]+'\n'+report[b:]
    marker='9. **Strict observed latency-and-energy dominance over both fixed routes:**'
    a=report.index(marker);b=report.index('\n10.',a)
    if 'Latency ties' not in report[a:b]:
        report=(report[:b]+' Latency ties within 1 ns also give W3 a lower-energy, no-worse-p95 comparison against both controls at two and four robots. '
            'These are observed point-estimate comparisons; control_dominance.csv records the tie rule.\n'+report[b:])
    marker='## Interpretation and robustness'
    a=report.index(marker)+len(marker)
    explanation=['','',
        'For W1, unavailable backhaul accounts for 6.00–7.23% of admissions and 11.81–13.58% of UAV assignments across populations. '
        'Most UAV assignments occur while backhaul is available. The 310 matched admissions produce 51/142/45/58 placement changes under W2/W3/W4/W5, respectively, with radio inputs, priorities and previous assignments fixed. '
        'This demonstrates a direct objective effect. A unique decomposition of whole-mission effects remains limited by state feedback and fixed previous assignments.','',
        'W1’s mean radio queue contribution rises from 1.19 ms at two robots to 46.23 ms at 32 robots, alongside a mean observed-minus-predicted latency gap rising from 1.63 to 41.15 ms. '
        'The mean number of fixed previous jobs rises from 0.49 to 9.48. Cloud compute peaks at one simultaneous job in these W1 runs, while UAV peaks reach 24 at 32 robots. '
        'Radio queueing and the difference between admission CPU predictions and later processor sharing explain why snapshot latency and delivered-command latency diverge.','',
        'The all-UAV 32-robot control misses every deadline despite approximately 53.3% mission-time CPU utilization: the one-second control interval includes idle time after the 250 ms cutoff. '
        'Its 32 tasks require 3.2 billion cycles per round, or at least 533 ms of aggregate service at 6 GHz, even before radio overhead.','']
    if 'For W1, unavailable backhaul accounts for' not in report:
        report=report[:a]+'\n'.join(explanation)+report[a:]
    report=report.replace('Total study wall time:', 'Primary simulation/reproduction/report workflow wall time:')
    (root/'report.md').write_text(report,encoding='utf-8')
    target=folder/'postprocessing'/Path(__file__).name;shutil.copy2(__file__,target)
    (folder/'report_review.json').write_text(json.dumps(dict(status='reviewed',matrix_simulations=sum(int(r['cases']) for r in summary),
        task_records=sum(int(r['tasks']) for r in summary),
        source_sha256=sha256(__file__),simulation_changes=0,solver_changes=0,
        findings='Mixed trade-off; no sharp UAV-to-cloud transition; W2 descriptive compromise; small exploratory scale cohorts'),indent=2)+'\n')
    return dict(status='reviewed',dominance_comparisons=len(dominance),paired_effect_rows=len(excerpt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    print(json.dumps(finalize(args.root)))
