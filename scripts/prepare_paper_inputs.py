"""Verify and archive supplied study inputs; never edits the source assets."""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from panama_uav.config import load_config
from panama_uav.paper_audit import association_average, audit_physics
from panama_uav.records import sha256, write_json
from panama_uav.training import load_model


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--timeline", required=True)
    args = parser.parse_args()
    pairs = [(Path(args.checkpoint).resolve(), ROOT/"artifacts/inputs/final_model.pt"),
             (Path(args.timeline).resolve(), ROOT/"data/leo/tremblant_full.npz")]
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(source) != sha256(target):
                raise ValueError(f"Existing input differs: {target}")
        else:
            shutil.copyfile(source, target)
    config = load_config(ROOT/"configs/paper_study.toml")
    _, payload = load_model(pairs[0][1], config, torch.device("cpu"), require_qualified=True)
    average = ROOT/"data/leo/tremblant_association_average.npz"
    if not average.exists():
        association_average(pairs[1][1], average)
    audit_physics(config, pairs[1][1], average, ROOT/"research/physics_audit.json")
    manifest = dict(checkpoint=dict(source=str(pairs[0][0]), local=str(pairs[0][1].relative_to(ROOT)),
        sha256=sha256(pairs[0][1]), qualification=payload["qualification"], map_compatibility=payload["map_compatibility"]),
        timeline=dict(source=str(pairs[1][0]), local=str(pairs[1][1].relative_to(ROOT)), sha256=sha256(pairs[1][1])),
        association_average=dict(local=str(average.relative_to(ROOT)), sha256=sha256(average)))
    write_json(ROOT/"research/assets_manifest.json", manifest)
    print("Qualified assets and physical assumptions verified.")


if __name__ == "__main__":
    main()
