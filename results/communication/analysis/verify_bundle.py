"""Verify published communication artifacts without local checkpoints or replays."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root):
    root = Path(root).resolve()
    with (root.parent / "files.csv").open(encoding="utf-8", newline="") as stream:
        entries = [row for row in csv.DictReader(stream) if row["study"] == "communication"]
    assert len(entries) == 43
    for entry in entries:
        path = root.parent / entry["file"]
        with path.open("rb") as stream:
            assert digest(stream) == entry["sha256"], path
        assert path.stat().st_size < 100 * 1024 * 1024, path
        if entry["format"].startswith("gzip"):
            with gzip.open(path, "rb") as stream:
                assert digest(stream) == entry["source_sha256"], path

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    cases = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    case_index = {row["case_id"]: row for row in cases}
    assert len(case_index) == len(cases) == 40
    expected = {(condition["name"], case_id) for condition in manifest["conditions"] for case_id in case_index}
    observed = set()
    with (root / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = (row["condition"], row["case_id"])
            assert identity in expected and identity not in observed, identity
            observed.add(identity)
            case = case_index[row["case_id"]]
            for name in ("num_agents", "seed"):
                assert int(row[name]) == case[name], (identity, name)
            for name in ("starts", "goals"):
                assert json.loads(row[name]) == case[name], (identity, name)
            assert row["map_sha256"] == case["map_sha256"], identity
            assert float(row["orbital_offset_seconds"]) == case["orbital_offset_seconds"], identity
    assert observed == expected
    counts = {"episodes": len(observed)}
    for name, target in (("tasks", validation["task_records"]), ("rounds", validation["round_records"])):
        with gzip.open(root / (name + ".csv.gz"), "rt", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            width = len(header)
            count = 0
            for row in reader:
                assert len(row) == width, (name, count)
                count += 1
        assert count == target, name
        counts[name] = count
    with gzip.open(root / "audits.jsonl.gz", "rt", encoding="utf-8") as stream:
        audits = sum(1 for line in stream if line.strip())
    assert audits == validation["audited_archives"] == len(observed)
    suite = ET.parse(root / "tests.xml").getroot()
    if suite.tag == "testsuites":
        suite = suite.find("testsuite")
    assert int(suite.attrib["tests"]) == 360
    assert int(suite.attrib["errors"]) == int(suite.attrib["failures"]) == 0
    return dict(status="passed", source_artifacts=len(entries),
                published_hashes_verified=True, gzip_source_bytes_verified=True,
                paired_case_identity_verified=True, csv_widths_verified=True,
                record_counts=counts, replay_audit_records=audits,
                source_regression_tests=360, largest_file_bytes=max(
                    p.stat().st_size for p in root.rglob("*") if p.is_file()))


if __name__ == "__main__":
    result = verify(Path(__file__).resolve().parents[1])
    print(json.dumps(result, indent=2))
