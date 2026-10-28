"""Generate a few communication_effects archives (N=8) through the repo's own archive path, for validating
prior_aoi_extract.py. Uses prepare() (copies SANGIN-dev/src to <root>/code) then _initialize/_episode from the
frozen copy for a handful of conditions only. No repo edits; outputs only under --root."""
import argparse, json, sys
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--root',required=True);ap.add_argument('--model',default='/workspace/panama_models/model8.pt')
ap.add_argument('--conditions',nargs='+',default=['ideal','allocated__nominal','cloud__nominal'])
ap.add_argument('--cap',type=int,default=250);ap.add_argument('--cases',type=int,default=1);ap.add_argument('--case-ids',nargs='*',default=[])
a=ap.parse_args();root=Path(a.root).resolve()
if not (root/'manifest.json').exists():
    sys.path.insert(0,'/workspace/SANGIN-dev/src')
    from panama_uav.config import load_config
    from panama_uav import communication_effects as CE
    cfg=load_config(Path('/workspace/SANGIN-dev/configs/paper_remote_dt_250ms.toml'))
    from dataclasses import replace
    cfg=replace(cfg,communication=replace(cfg.communication,leo_timeline='/workspace/leo_study/inputs/tremblant_full.npz'))
    CE.prepare(root,cfg,Path(a.model),cases=a.cases,counts=(8,),cap=a.cap,workers=1,diagnostic=True,budget_seconds=7200)
    print('prepared',root,flush=True)
    import subprocess,os
    env=dict(os.environ,PYTHONPATH=str(root/'code'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    sys.exit(subprocess.call([sys.executable,__file__,'--root',str(root),'--cap',str(a.cap),'--case-ids',*a.case_ids,'--conditions',*a.conditions],env=env))
from panama_uav import communication_effects as CE
assert Path(CE.__file__).resolve().is_relative_to(root/'code'),CE.__file__
CE._initialize(str(root))
m=json.loads((root/'manifest.json').read_text());bank=json.loads((root/'paired_cases.json').read_text())
conds={c['name']:c for c in m['conditions']}
for name in a.conditions:
    for case in bank:
        if a.case_ids and case['case_id'] not in a.case_ids: continue
        row,tasks,rounds,audit=CE._episode((conds[name],case,m['episode_cap']))
        print(json.dumps(dict(condition=name,case=case['case_id'],success=row.get('success'),steps=row.get('steps'),
            **{k:audit.get(k) for k in ('status','causal_events','ordering_pairs_verified','peer_shadows_rebuilt','wall_seconds')})),flush=True)
        with (root/'val_audits.jsonl').open('a') as f:f.write(json.dumps(audit)+'\n')
