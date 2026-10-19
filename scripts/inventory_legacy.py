#!/usr/bin/env python3
"""Inventory the repository and optionally move the superseded implementation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


LEGACY_TARGETS = (
    "AerialBaseStation.py",
    "config_parallel.py",
    "environment.py",
    "infer.py",
    "infer_UAV.py",
    "main_parallel.py",
    "models.py",
    "rl_module.py",
    "utils.py",
    "Models",
    "Checkpoints",
    "Metrics",
    "Test",
    "solutions",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category(relative: Path) -> str:
    first = relative.parts[0]
    if first in {"src", "configs", "scripts", "tests", "docs", "maps", "scens"}:
        return "active"
    if first == "artifacts":
        return "artifact"
    if first in LEGACY_TARGETS or relative.name in LEGACY_TARGETS:
        return "legacy"
    if "__pycache__" in relative.parts or relative.suffix == ".pyc" or relative.name == ".DS_Store":
        return "cache"
    return "project"


def inventory(root: Path) -> list[dict[str, object]]:
    paths = [path for path in root.rglob("*") if path.is_file() and "artifacts/inventory" not in path.as_posix()]
    records: list[dict[str, object]] = []
    for path in sorted(paths):
        relative = path.relative_to(root)
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "category": category(relative),
            }
        )
    return records


def write_inventory(root: Path, records: list[dict[str, object]]) -> Path:
    destination = root / "artifacts" / "inventory"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "repository_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "size_bytes", "sha256", "category"))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": len(records),
        "bytes": sum(int(record["size_bytes"]) for record in records),
        "categories": {
            name: sum(1 for record in records if record["category"] == name)
            for name in sorted({str(record["category"]) for record in records})
        },
        "inventory_csv": str(csv_path.relative_to(root)),
    }
    (destination / "repository_inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path


def archive_legacy(root: Path, label: str) -> list[str]:
    destination = root / "artifacts" / "legacy" / label
    destination.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for relative_name in LEGACY_TARGETS:
        source = root / relative_name
        if not source.exists():
            continue
        target = destination / relative_name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing archive target: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append(relative_name)
    (destination / "ARCHIVE.json").write_text(
        json.dumps(
            {
                "archived_at_utc": datetime.now(timezone.utc).isoformat(),
                "recoverable": True,
                "original_root": str(root),
                "moved": moved,
                "inventory": "../../inventory/repository_inventory.csv",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return moved


def clean_caches(root: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() and (path.name == ".DS_Store" or path.suffix == ".pyc" or path.name.endswith(".tmp")):
            removed.append(path.relative_to(root).as_posix())
            path.unlink()
        elif path.is_dir() and path.name == "__pycache__":
            shutil.rmtree(path)
            removed.append(path.relative_to(root).as_posix() + "/")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--clean-caches", action="store_true")
    parser.add_argument("--label", default=datetime.now(timezone.utc).date().isoformat())
    args = parser.parse_args()
    root = args.root.resolve()
    records = inventory(root)
    manifest = write_inventory(root, records)
    result: dict[str, object] = {"inventory": str(manifest), "files": len(records)}
    if args.archive:
        result["archived"] = archive_legacy(root, args.label)
    if args.clean_caches:
        result["removed_caches"] = clean_caches(root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

