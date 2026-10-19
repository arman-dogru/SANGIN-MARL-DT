"""Resume an unchanged saved runner with transient Windows metadata-write retries.

Only the runner's JSON metadata writer is wrapped. OS waits here are for
file sharing locks; they never advance the communication simulator's clock.
Completed episodes are skipped by the original runner's case-ID resume logic.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    frozen = args.runner.resolve()
    spec = importlib.util.spec_from_file_location("frozen_inference_runner",frozen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.write_json
    def retry_metadata(path,value):
        for attempt in range(24):
            try:
                return original(path,value)
            except PermissionError:
                if attempt == 23:raise
                delay = min(.5,.05*2**min(attempt,4))
                with (frozen.parent/"io_recovery.jsonl").open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps({"event":"metadata_write_retry",
                        "time_utc":datetime.now(timezone.utc).isoformat(),
                        "path":str(path),"attempt":attempt+1,"wall_wait_seconds":delay})+'\n')
                time.sleep(delay)
    module.write_json = retry_metadata
    sys.argv = [str(frozen),*forwarded]
    module.main()


if __name__ == "__main__":
    main()
