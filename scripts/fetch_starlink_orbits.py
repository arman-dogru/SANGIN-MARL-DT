"""Archive real orbital elements; never silently substitute a generated shell."""

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/leo")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = [
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json",
        "https://raw.githubusercontent.com/satvisorcom/satvisor-data/master/celestrak/json/starlink.json",
    ]
    errors = []
    for url in candidates:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read()
            records = json.loads(raw)
            if (
                not isinstance(records, list)
                or not records
                or "EPOCH" not in records[0]
            ):
                raise ValueError("Not an OMM array")
            break
        except (OSError, ValueError, KeyError) as error:
            errors.append({"url": url, "error": str(error)})
    else:
        raise RuntimeError(json.dumps(errors))
    metadata = {
        "url": url,
        "original_source": "CelesTrak",
        "mirror": url != candidates[0],
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "elements": len(records),
        "epoch_min": min(r["EPOCH"] for r in records),
        "epoch_max": max(r["EPOCH"] for r in records),
        "failed_sources": errors,
    }
    # Pin a mirror revision when its bytes match the downloaded snapshot.
    if metadata["mirror"]:
        try:
            query = "https://api.github.com/repos/satvisorcom/satvisor-data/commits?path=celestrak/json/starlink.json&per_page=1"
            with urllib.request.urlopen(query, timeout=30) as response:
                revision = json.load(response)[0]["sha"]
            pinned = url.replace("/master/", f"/{revision}/")
            with urllib.request.urlopen(pinned, timeout=30) as response:
                pinned_raw = response.read()
            if hashlib.sha256(pinned_raw).hexdigest() == metadata["source_sha256"]:
                metadata["immutable_url"] = pinned
                metadata["mirror_revision"] = revision
        except (OSError, ValueError, KeyError, IndexError) as error:
            metadata["revision_lookup_error"] = str(error)
    (output / "starlink_omm.json").write_bytes(raw)
    (output / "orbit_source.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
