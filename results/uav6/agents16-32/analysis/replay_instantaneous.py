"""Reproduce and verify the supplementary instantaneous MAPF reference."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from panama_uav.config import ExperimentMode, ProjectConfig
from panama_uav.models import select_action, select_actions
from panama_uav.phase1 import clean_config
from panama_uav.phase1_execution import RemoteAlignedCleanMAPFEnvironment
from panama_uav.records import sha256, write_json
from panama_uav.scenarios import Scenario
from panama_uav.training import load_model


def main(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    expected = json.loads((root/'mapf_reference_serial.json').read_text())
    config = ProjectConfig.from_dict(manifest['configuration'])
    assert sha256(root/'assets/checkpoint.pt') == manifest['model_sha256']
    assert sha256(root/'cases.json') == expected['case_bank_sha256']
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    device = torch.device('cpu')
    model, _ = load_model(root/'assets/checkpoint.pt', config, device, require_qualified=True)
    config = clean_config(config)
    records, started = [], time.monotonic()
    for case in json.loads((root/'cases.json').read_text()):
        scenario = Scenario(Path(case['case_id']), Path(case['map_path']),
                            tuple(map(tuple,case['starts'])),tuple(map(tuple,case['goals'])))
        environment = RemoteAlignedCleanMAPFEnvironment(config, ExperimentMode.UAV,
            case['map_path'], case['num_agents'], 250, case['seed'], scenario)
        observations, _ = environment.reset()
        rng = np.random.default_rng(case['seed'])
        comparisons = disagreements = 0
        while True:
            active = sorted(environment.active_robot_ids)
            actual = {robot:select_action(model,observations[robot],0.,rng,device) for robot in active}
            batched = select_actions({0:model},{robot:0 for robot in active},
                {robot:observations[robot] for robot in active},0.,rng,device)
            comparisons += len(active)
            disagreements += sum(actual[robot] != batched[robot] for robot in active)
            result = environment.step(actual)
            observations = result.observations
            environment.communication.transport.log.clear()
            if result.terminated or result.truncated:
                break
        record = dict(case_id=case['case_id'],num_agents=case['num_agents'],seed=case['seed'],
            steps=environment.step_count,success=result.terminated,
            inference_comparisons=comparisons,batch_serial_disagreements=disagreements)
        assert record == next(r for r in expected['records'] if r['case_id']==case['case_id']), record
        records.append(record)
    result = dict(status='passed',case_count=len(records),records=records,
        inference_comparisons=sum(r['inference_comparisons'] for r in records),
        batch_serial_disagreements=sum(r['batch_serial_disagreements'] for r in records),
        model_sha256=manifest['model_sha256'],case_bank_sha256=sha256(root/'cases.json'),
        reference_sha256=sha256(root/'mapf_reference_serial.json'),script_sha256=sha256(__file__),
        wall_seconds=time.monotonic()-started,policy_updates=0)
    assert result['inference_comparisons']==expected['inference_comparisons']
    write_json(root/'analysis/reference_reproduction.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().root)
