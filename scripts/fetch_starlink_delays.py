"""Fetch paired recordings using HTTP ranges, preserving dataset provenance."""

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from panama_uav.transport import import_trace

URL = "https://zenodo.org/api/records/16275284/files/StarlinkOWD.zip/content"


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.downloaded_bytes = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (
            self.position if whence == 1 else self.size if whence == 2 else 0
        )
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(self.size - self.position, self.size if size < 0 else size)
        if size <= 0:
            return b""
        start = self.position
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{start + size - 1}"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206 or not response.headers.get(
                "Content-Range", ""
            ).startswith(f"bytes {start}-"):
                raise ValueError("Server did not honor the requested archive range")
            data = response.read()
        if len(data) != size:
            raise ValueError("Truncated archive range")
        self.position += len(data)
        self.downloaded_bytes += len(data)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/leo/starlink_owd")
    parser.add_argument("--pair-index", type=int, default=0)
    args = parser.parse_args()
    output = PROJECT / args.output
    output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        "https://zenodo.org/api/records/16275284", timeout=60
    ) as response:
        metadata = json.load(response)
    archive = next(f for f in metadata["files"] if f["key"] == "StarlinkOWD.zip")
    stream = RangeFile(URL, archive["size"])
    files = []
    with zipfile.ZipFile(stream) as remote:
        names = remote.namelist()
        uplinks = sorted(name for name in names if name.endswith("__ul.csv.zip"))
        pairs = [
            (u, u.replace("__ul.csv.zip", "__dl.csv.zip"))
            for u in uplinks
            if u.replace("__ul.csv.zip", "__dl.csv.zip") in names
        ]
        if not pairs:
            raise ValueError("No paired recordings found")
        for direction, name in zip(("uplink", "downlink"), pairs[args.pair_index]):
            zipped = remote.read(name)  # ZipFile verifies this member's CRC.
            (output / name).write_bytes(zipped)
            with zipfile.ZipFile(io.BytesIO(zipped)) as inner:
                csv_name = next(n for n in inner.namelist() if n.endswith(".csv"))
                csv = inner.read(csv_name)
            path = output / Path(csv_name).name
            path.write_bytes(csv)
            imported = import_trace(
                path, output / f"{direction}.npz", direction, "ns", origin_seconds=0.0
            )
            files.append(
                {
                    "direction": direction,
                    "archive_member": name,
                    "member_sha256": hashlib.sha256(zipped).hexdigest(),
                    "csv_sha256": hashlib.sha256(csv).hexdigest(),
                    "import": imported,
                }
            )
            print(json.dumps(files[-1]), flush=True)
    manifest = {
        "dataset": "https://zenodo.org/records/16275284",
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "measurement_site": "Karlstad, Sweden",
        "measurement_month": "2025-04",
        "usage": "empirical transport sensitivity; not a Mont-Tremblant measurement",
        "download_method": "HTTP range extraction with per-member ZIP CRC and SHA-256",
        "archive_declared_checksum": archive["checksum"],
        "full_archive_checksum_verified": False,
        "downloaded_bytes": stream.downloaded_bytes,
        "archive_bytes": archive["size"],
        "paired_recordings_available": len(pairs),
        "selected_pair_index": args.pair_index,
        "missing_sequence_numbers_are_loss": False,
        "handovers_inferred_from_spikes": False,
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
