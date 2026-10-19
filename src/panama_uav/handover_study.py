"""Explicit inference ablations of archived satellite-change interruptions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


def derive_timeline(source, output, stress_seconds=0.0):
    """Preserve geometry; separate resource reconfiguration from satellite handover.

    Causes: 0 = visibility gap, 1 = satellite change, 2 = other setup/reconfiguration.
    A positive stress duration replaces only actual satellite-change interruption
    durations. Overlapping outages are unioned, including the periodic replay seam.
    """
    if stress_seconds < 0 or not np.isfinite(stress_seconds):
        raise ValueError("Stress duration must be finite and nonnegative")
    source, output = Path(source), Path(output)
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    times = arrays["times"]
    first, last = float(times[0]), float(times[-1])
    period = last-first
    events = deepcopy(json.loads(str(arrays["handover_events"])))
    by_start = {event["start"]: event for event in events}
    windows = []
    for left, right, cause in arrays["blocked_windows"]:
        left, right = float(left), float(right)
        if cause == 1:
            event = by_start[left]
            change = bool(event.get("attachment_changed", event["from_satellite"] != event["to_satellite"]))
            change = change and not event.get("initial_attachment", False)
            category = "initial_attachment" if event.get("initial_attachment", False) else "satellite_change" if change else "resource_reconfiguration"
            event.update(service_category=category,
                         original_end=right, artificial_stress_seconds=stress_seconds if change else 0.0)
            cause = 1 if change else 2
            if change and stress_seconds:
                right = max(right, left+stress_seconds)
                event["end"] = right
                if right > event["original_end"]:
                    event.setdefault("control_messages", []).append({
                        "kind": "artificial_service_outage", "start": event["original_end"], "end": right})
        if right-left >= period:
            raise ValueError("Stress would remove all satellite service in the replay period")
        windows.append((left, min(right, last), int(cause)))
        if right > last:
            windows.append((first, first+right-last, int(cause)))
    windows.sort()
    merged = []
    for left, right, cause in windows:
        if merged and left <= merged[-1][1]:
            previous = merged.pop()
            # Mixed-category overlap occurs only in stress timelines, which
            # require interruptions enabled. Union time is charged exactly once.
            kind = 0 if 0 in (previous[2], cause) else 2 if 2 in (previous[2], cause) else 1
            merged.append((previous[0], max(previous[1], right), kind))
        else:
            merged.append((left, right, cause))
    blocked_seconds = sum(right-left for left, right, _ in merged)
    if blocked_seconds >= period-1e-9:
        raise ValueError("No satellite service remains in the replay period")
    metadata = json.loads(str(arrays["metadata"]))
    metadata.update(handover_study=True, stress_handover_seconds=stress_seconds,
        parent_timeline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        window_categories={"0": "visibility_gap", "1": "satellite_change", "2": "resource_reconfiguration"},
        control_semantics="handover_enabled toggles satellite-change interruptions only; resource reconfiguration stays active",
        stress_is_measured=False, blocked_seconds_per_period=blocked_seconds,
        overlapping_windows_unioned=True)
    arrays.update(blocked_windows=np.asarray(merged, dtype=float).reshape(-1, 3),
                  handover_events=np.array(json.dumps(events)), metadata=np.array(json.dumps(metadata)))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return metadata
