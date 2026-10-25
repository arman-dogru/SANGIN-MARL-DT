from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'code'))
import run_omega_sensitivity as driver
from panama_uav.records import sha256
revision=json.loads((root/'validation_revision.json').read_text())
assert sha256(root/'manifest.json')==revision['original_manifest_sha256']
for name,digest in revision['files'].items():
    assert sha256(Path(__file__).parent/name)==digest
from audit_omega_reproduction import baseline_gate
driver.baseline_gate=baseline_gate
if __name__=='__main__':driver.run(root)
