"""Audit and edit measured 100 ms deadline / handover inference figures."""
import argparse
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

FIELDS = ('task','intentions','priorities','all')
GROUPS = ('handover_off','handover_on','stress_10s','stress_20s')
LABELS = dict(task='A* guidance',intentions='Intentions',priorities='Priorities',all='All three')
COLORS = dict(task='#2563b5',intentions='#d96b20',priorities='#8b5abf',all='#cb3348')
MARKERS = dict(task='o',intentions='s',priorities='^',all='x')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return [json.loads(s) for s in path.read_text().splitlines() if s]


def export(path, rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def audit(root):
    plan=json.loads((root/'plan.json').read_text())
    state=json.loads((root/'status.json').read_text())
    assert state['status']=='complete' and state['completed_episodes']==4200
    assert sha(root/'baseline.pt')==plan['baseline_model_sha256']
    assert sha(root/'held_out_cases.json')==plan['case_bank_sha256']
    assert sha(root/'assets/maps/random-32-32-10.map')==plan['map_sha256']
    assert {p.name:sha(p) for p in (root/'code/panama_uav').glob('*.py')}==plan['code_sha256']
    assert all(sha(root/'assets'/n)==h for n,h in plan['timeline_sha256'].items())
    assert all(sha(root/'code'/n)==h for n,h in plan['script_sha256'].items())
    specs={s['name']:s for s in plan['conditions']}
    for field in FIELDS:
        off=specs[f'handover_off_{field}']['transport']
        on=specs[f'handover_on_{field}']['transport']
        assert {key for key in off if off[key]!=on[key]}=={'handover_enabled'}
    with np.load(root/'assets/orbit.npz',allow_pickle=False) as raw:
        for name in ('native','stress10','stress20'):
            with np.load(root/f'assets/orbit_{name}.npz',allow_pickle=False) as derived:
                for key in raw.files:
                    if key not in ('metadata','handover_events','blocked_windows'):
                        assert np.array_equal(raw[key],derived[key],equal_nan=True)
    bank={r['case_id']:r for r in json.loads((root/'held_out_cases.json').read_text())}
    all_rows, by_condition=[],{}
    summary={(r['condition'],r['stage']):r for r in json.loads((root/'summary.json').read_text())}
    for spec in plan['conditions']:
        name=spec['name']
        rows=read(root/name/'episodes.jsonl')
        assert len(rows)==200 and {r['case_id'] for r in rows}==set(bank)
        for row in rows:
            assert row['diagnostic'] is False
            assert all(row[k]==bank[row['case_id']][k] for k in ('seed','starts','goals','num_agents'))
            assert row['policy_model_sha256']==plan['baseline_model_sha256']
            assert row['stale_information']==spec['transport']['stale_information']
            assert row['decision_deadline_seconds']==.1 and 0<row['steps']<=250
            assert row['par2']==(row['steps'] if row['success'] else 500)
            for field in FIELDS[:3]:
                if field not in row['stale_information']:
                    assert row[f'decision_max_{field}_age_seconds']==0
            if row['group']=='handover_off': assert row['satellite_interruption_events_applied']==0
        for stage in range(1,5):
            group=[r for r in rows if r['stage']==stage]
            assert len(group)==50
            s=summary[name,stage]
            assert np.isclose(s['success_rate'],np.mean([r['success'] for r in group]),rtol=0,atol=1e-12)
            assert np.isclose(s['par2'],np.mean([r['par2'] for r in group]),rtol=0,atol=1e-12)
            solved=[r['steps'] for r in group if r['success']]
            assert (s['successful_makespan'] is None and not solved) or np.isclose(s['successful_makespan'],np.mean(solved),rtol=0,atol=1e-12)
            with gzip.open(root/name/f"{group[0]['case_id']}.json.gz",'rt') as f: trace=json.load(f)
            assert trace['frames'][0]['positions']==group[0]['starts'] and trace['goals']==group[0]['goals']
            decisions=[f['decision'] for f in trace['frames'] if f['decision'] is not None]
            assert len(decisions)==group[0]['steps']
            assert all(abs(d['at']-d['cycle_start']-.1)<1e-9 for d in decisions)
            assert all(v['packet_source_time']<=d['at']+1e-9 for d in decisions
                       for values in d['inputs'].values() for v in values.values())
            assert len({v for f in trace['frames'] for v in f['model_versions'].values()})==1
        by_condition[name]={r['case_id']:r for r in rows}
        all_rows.extend(rows)
    reference=by_condition['perfect']
    old={r['case_id']:r for r in read(Path(plan['previous_root'])/'perfect/episodes.jsonl')}
    for case,row in reference.items():
        assert all(row[k]==old[case][k] for k in ('success','steps','par2','collision_attempts','uav_travel_m'))
    for field in FIELDS:
        for case,row in by_condition[f'edge_control_{field}'].items():
            assert all(row[k]==reference[case][k] for k in ('success','steps','par2','collision_attempts'))
    assert len(all_rows)==4200
    return dict(status='passed',episodes=4200,conditions=21,cases_per_stage=50,
        identical_seeds_starts_goals=True, frozen_policy=True, decision_phase_verified=True,
        model_map_timeline_and_script_hashes_verified=True,geometry_and_rates_preserved=True,
        handover_toggle_isolated=True,
        unselected_inputs_fresh=True, summary_values_recomputed=True,
        perfect_matches_previous_run=True, uav_edge_matches_perfect_robot_outcomes=True), all_rows


def save(fig, output, name):
    for ext in ('png','pdf','svg'): fig.savefig(output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def figures(rows, output):
    lookup={(r['group'],r['fields'],r['stage']):r for r in rows}
    result=[]
    metrics=(('success_rate','Success (%)',100),('successful_makespan','Successful makespan (steps)',1),
             ('par2','Failure-aware PAR-2 (steps)',1),('deadline_miss_fraction','Missed response deadlines (%)',100),
             ('selected_input_mean_age_seconds','Mean episode input age (s)',1))
    for metric,label,scale in metrics:
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        fig.subplots_adjust(left=.085,right=.985,bottom=.085,top=.84,wspace=.23,hspace=.38)
        for stage,ax in enumerate(axes.flat,1):
            for field in FIELDS:
                values=[lookup[group,field,stage][metric] for group in GROUPS]
                ax.plot(range(4),[v*scale if v is not None else np.nan for v in values],
                        color=COLORS[field],marker=MARKERS[field],lw=1.7,ms=6,label=LABELS[field])
            perfect=lookup['perfect','perfect',stage][metric]
            if perfect is not None: ax.axhline(perfect*scale,color='#334155',ls='--',lw=1.5)
            ax.set(title=f'Stage {stage}',xticks=range(4),
                xticklabels=['Handovers\noff','Handovers\non','10 s\nstress','20 s\nstress'],ylabel=label)
            if metric in ('success_rate','deadline_miss_fraction'): ax.set_ylim(0,100)
            ax.grid(alpha=.18)
        handles=[Line2D([],[],color='#334155',ls='--',label='Perfect communication')]
        handles += [Line2D([],[],color=COLORS[f],marker=MARKERS[f],label=LABELS[f]) for f in FIELDS]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.947),ncol=3,fontsize=10)
        fig.suptitle('Cloud mission processing · 100 ms decision deadline',y=.995,fontsize=14)
        save(fig,output,metric)
        result.append(metric)
    return result


def latency_timeline(root, output):
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for group,ax in zip(GROUPS,axes.flat):
        path=root/f'{group}_all/agents-8-case-000.json.gz'
        with gzip.open(path,'rt') as f: trace=json.load(f)
        delivered=[e for e in trace['transport_events'] if e['event']=='delivered' and e['kind']=='information']
        ax.scatter([e['source_time'] for e in delivered],[e['age_seconds']*1000 for e in delivered],s=7,alpha=.35)
        ax.axhline(100,color='#cb3348',ls='--',lw=1.5,label='100 ms deadline')
        merged=[]
        for left,right in sorted(trace['handover_windows']):
            if merged and left<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],right)
            else: merged.append([left,right])
        for left,right in merged:
            ax.axvspan(max(0,left),min(trace['frames'][-1]['time'],right),color='#2563b5',alpha=.09)
        for event in trace['transport_events']:
            if event['event']=='orbit_handover' and event['source_window']['service_category']=='satellite_change':
                w=event['source_window']
                at=event['time']
                ax.axvline(at,color='#334155',lw=.5,alpha=.4)
        ax.set(title=group.replace('_',' '),xlabel='Report generation time (s)',ylabel='Delivered update latency (ms)',yscale='log')
        ax.set_xlim(0,trace['frames'][-1]['time'])
        ax.grid(alpha=.18)
        ax.legend(fontsize=9,loc='upper right',frameon=True,facecolor='white',edgecolor='none',framealpha=.9)
    fig.suptitle('One paired Stage 4 case · change markers and shaded service interruptions',fontsize=14)
    save(fig,output,'latency_timeline')
    return 'latency_timeline'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    root=args.root.resolve()
    output=args.output or root/'plots'
    output.mkdir(parents=True,exist_ok=True)
    checked,episodes=audit(root)
    rows=json.loads((root/'summary.json').read_text())
    export(output/'summary.csv',rows)
    export(output/'episodes.csv',episodes)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','legend.frameon':False})
    names=figures(rows,output)
    names.append(latency_timeline(root,output))
    script=Path(__file__)
    if script.resolve()!=(output/script.name).resolve(): shutil.copy2(script,output/script.name)
    metadata=dict(audit=checked,figures=names,source_root=str(root),
        model_sha256=sha(root/'baseline.pt'),postprocessing_script_sha256=sha(Path(__file__)),synthetic_extension=False,
        caveats=['10/20-second interruptions are artificial stresses at actual satellite changes, not measured LEO durations',
                 'Handovers off retains resource reconfiguration and all ordinary transport delays',
                 'Action cadence remains one per second; deadline is a fixed 100 ms phase within the cycle',
                 'Makespan averages successful episodes; PAR-2 assigns 500 to failures',
                 'Paired confidence intervals quantify case variability for one frozen training seed'])
    (root/'audit.json').write_text(json.dumps(checked,indent=2))
    (output/'results.json').write_text(json.dumps(metadata,indent=2))
    blocks=['<!doctype html><meta charset="utf-8"><title>Decision deadline and handovers</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 20px}img{width:100%}figure{margin:40px 0}p{line-height:1.6}</style><h1>100 ms decision deadline and satellite handovers</h1><p>50 paired cases per stage, one qualified frozen baseline. Satellite-change interruptions are switched off/on; resource reconfiguration and ordinary transport stay fixed. The 10/20-second durations are explicit artificial stress tests.</p>']
    for name in names: blocks.append(f'<figure><img src="{name}.png"><figcaption>{html.escape(name.replace("_"," "))} · <a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></figcaption></figure>')
    blocks.append('<p>Successful makespan excludes failures. PAR-2 retains them as 500 steps. Per-stage summaries include paired comparisons against both Perfect and handovers off. Messages arriving after a deadline remain timestamped and can be used at a later decision.</p>')
    (output/'index.html').write_text('\n'.join(blocks),encoding='utf-8')
    with zipfile.ZipFile(root/'deadline_inference_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file(): archive.write(path,path.name)
    print(json.dumps(dict(plots=str(output),audit=checked,figures=names),indent=2))


if __name__=='__main__': main()
