"""Record passing regression evidence and the exact source used for preparation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from panama_uav.records import sha256


def junit_counts(path):
    path=Path(path).resolve();root=ET.parse(path).getroot()
    suites=[root] if root.tag=='testsuite' else root.findall('testsuite')
    result={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if not result['tests'] or result['failures'] or result['errors']:
        raise ValueError('Passing, nonempty test results are required: '+str(path))
    return dict(result,path=str(path),sha256=sha256(path))


def record(full,focused,destination):
    project=Path(__file__).resolve().parents[1]
    full,focused=junit_counts(full),junit_counts(focused)
    paths=list((project/'src/panama_uav').glob('*.py'))+list((project/'tests').glob('*.py'))
    paths += [project/'scripts'/n for n in ('run_redesign.py','report_redesign.py',
        'run_omega_sensitivity.py','audit_omega_reproduction.py','record_redesign_validation.py')]
    result=dict(status='passed',protocol='dt_bundle_slow_pipeline_v2',recorded_utc=datetime.now(timezone.utc).isoformat(),
        full_suite=full,focused_final_changes=focused,full_suite_failures=full['failures'],full_suite_errors=full['errors'],
        source_sha256={p.relative_to(project).as_posix():sha256(p) for p in paths})
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',type=Path,required=True);parser.add_argument('--focused',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('tmp/redesign-validation.json'))
    args=parser.parse_args();result=record(args.full,args.focused,args.output)
    print(json.dumps(dict(status=result['status'],full_tests=result['full_suite']['tests'],focused_tests=result['focused_final_changes']['tests'])))
