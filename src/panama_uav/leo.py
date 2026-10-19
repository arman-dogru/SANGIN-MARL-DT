"""Data-driven orbital backhaul timeline; no provider scheduler is implied."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

C_KM_S = 299_792.458
EARTH_RATE = 7.29211514670698e-5


def observer_ecef(latitude, longitude, altitude_km):
    lat, lon = np.radians([latitude, longitude])
    a, e2 = 6378.137, 6.69437999014e-3
    radius = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    position = np.array(
        [
            (radius + altitude_km) * np.cos(lat) * np.cos(lon),
            (radius + altitude_km) * np.cos(lat) * np.sin(lon),
            (radius * (1 - e2) + altitude_km) * np.sin(lat),
        ]
    )
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return position, up


def teme_to_ecef(position, velocity, julian_dates):
    """Vallado GMST rotation, UTC approximates UT1; polar motion is omitted."""
    centuries = (np.asarray(julian_dates) - 2451545.0) / 36525.0
    seconds = (
        67310.54841
        + (876600 * 3600 + 8640184.812866) * centuries
        + 0.093104 * centuries**2
        - 6.2e-6 * centuries**3
    )
    angle = np.remainder(seconds * np.pi / 43200, 2 * np.pi)
    cosine, sine = np.cos(angle), np.sin(angle)
    r = np.stack(
        (
            cosine * position[..., 0] + sine * position[..., 1],
            -sine * position[..., 0] + cosine * position[..., 1],
            position[..., 2],
        ),
        axis=-1,
    )
    v = np.stack(
        (
            cosine * velocity[..., 0] + sine * velocity[..., 1],
            -sine * velocity[..., 0] + cosine * velocity[..., 1],
            velocity[..., 2],
        ),
        axis=-1,
    )
    v[..., 0] += EARTH_RATE * r[..., 1]
    v[..., 1] -= EARTH_RATE * r[..., 0]
    return r, v


def link_geometry(position, velocity, observer, up):
    line = position - observer
    distance = np.linalg.norm(line, axis=-1)
    unit = line / distance[..., None]
    elevation = np.degrees(np.arcsin(np.clip(unit @ up, -1, 1)))
    range_rate = np.sum(velocity * unit, axis=-1)
    return elevation, distance, range_rate


def handshake(
    at,
    propagation,
    uplink_mbps=10,
    downlink_mbps=50,
    synchronization_seconds=0.005,
    processing_seconds=0.005,
):
    """An explicit assumed four-message protocol, not measured Starlink signaling."""
    cursor = at + synchronization_seconds
    events = [{"kind": "synchronization", "start": at, "end": cursor}]
    for label, direction, size in (
        ("request", "uplink", 64),
        ("grant", "downlink", 128),
        ("complete", "uplink", 64),
        ("ack", "downlink", 128),
    ):
        capacity = uplink_mbps if direction == "uplink" else downlink_mbps
        start = cursor
        transmission = size * 8 / (capacity * 1e6)
        cursor += transmission + propagation
        events.append(
            {
                "kind": label,
                "direction": direction,
                "bytes": size,
                "start": start,
                "end": cursor,
                "transmission_seconds": transmission,
                "propagation_seconds": propagation,
            }
        )
        if label in ("request", "complete"):
            cursor += processing_seconds
    return cursor, events


@dataclass
class OrbitTimeline:
    times: np.ndarray
    propagation: np.ndarray
    blocked: np.ndarray
    events: list
    metadata: dict
    series: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            result = cls(
                data["times"],
                data["propagation_seconds"],
                data["blocked_windows"],
                json.loads(str(data["handover_events"])),
                json.loads(str(data["metadata"])),
                {name: data[name] for name in data.files if name not in (
                    "times", "propagation_seconds", "blocked_windows", "handover_events", "metadata"
                )},
            )
        if len(result.times) < 2 or np.any(np.diff(result.times) <= 0):
            raise ValueError("Orbit timeline must have ordered timestamps")
        if np.any(result.blocked[:, 1] < result.blocked[:, 0]) or (
            len(result.blocked) > 1
            and np.any(result.blocked[1:, 0] < result.blocked[:-1, 1])
        ):
            raise ValueError(
                "Orbit interruption windows must be ordered and non-overlapping"
            )
        return result

    @property
    def period(self):
        return float(self.times[-1] - self.times[0])

    def local_time(self, at, repeat=False):
        if repeat:
            return self.times[0] + (at - self.times[0]) % self.period
        if not self.times[0] <= at <= self.times[-1]:
            raise ValueError("Simulation exceeds orbital dataset timeline")
        return at

    def state(self, at, repeat=False):
        local = self.local_time(at, repeat)
        index = max(0, int(np.searchsorted(self.times, local, side="right")) - 1)
        result = {name: float(values[index]) for name, values in self.series.items()
                  if values.ndim == 1 and len(values) == len(self.times)}
        result.update(timeline_seconds=local, replay_cycle=int((at-self.times[0])//self.period) if repeat else 0)
        return result

    def events_between(self, start, end, repeat=False):
        if end < start:
            return []
        period = self.period
        cycles = range(int(start // period), int(end // period) + 1) if repeat else (0,)
        result = []
        starts = np.array([event["start"] for event in self.events])
        for cycle in cycles:
            shift = cycle * period
            left = np.searchsorted(starts, start-shift, side="right")
            right = np.searchsorted(starts, end-shift, side="right")
            for event in self.events[left:right]:
                item = dict(event, start=event["start"]+shift, end=event["end"]+shift,
                            replay_cycle=cycle)
                item["control_messages"] = [dict(control, start=control["start"]+shift,
                                                  end=control["end"]+shift)
                                             for control in event.get("control_messages", [])]
                result.append(item)
        return result

    def service(self, start, bits, link, ceiling_bps, frame_seconds, guard_fraction,
                repeat=False, handover_enabled=True):
        """Integrate time-varying service, frame guards and interruption windows.

        The framing/round-robin policy is an experiment assumption, not Starlink MAC.
        ITU-R and Shannon rates are precomputed by the library-backed builder.
        """
        cursor, remaining = float(start), float(bits)
        transmitting = blocked_time = guard_time = 0.0
        starts = self.blocked[:, 0]
        while remaining > 0:
            local = self.local_time(cursor, repeat)
            shift = cursor - local
            index = max(0, int(np.searchsorted(self.times, local, side="right"))-1)
            if index == len(self.times)-1:
                raise ValueError("Satellite service cannot finish within orbital timeline")
            candidate = int(np.searchsorted(starts, local, side="right"))-1
            # Replay subtraction can put local time just below a boundary that
            # cursor has already reached. Classify and advance with the same
            # absolute boundary, so an interruption cannot set cursor to itself.
            if candidate+1 < len(starts) and cursor >= float(starts[candidate+1])+shift:
                candidate += 1
            if candidate >= 0:
                left, right, cause = self.blocked[candidate]
                absolute_end = float(right)+shift
                if cursor < absolute_end and (cause != 1 or handover_enabled):
                    blocked_time += absolute_end-cursor
                    cursor = absolute_end
                    continue
            # A 20 ms frame reserves its final 5% for guard/control airtime.
            # Classify against representable boundary products, rather than an
            # epsilon on cursor/frame_seconds. At 32775.06 s that quotient
            # rounds below its integer; start+frame_seconds then equals cursor
            # and the old guard branch loops forever without serving any bits.
            frame_index = int(np.floor(cursor/frame_seconds))
            if frame_index*frame_seconds > cursor:
                frame_index -= 1
            frame_end = (frame_index+1)*frame_seconds
            if cursor >= frame_end:
                frame_index += 1
                frame_end = (frame_index+1)*frame_seconds
            frame_start = frame_index*frame_seconds
            data_end = frame_start+frame_seconds*(1-guard_fraction)
            if cursor >= data_end:
                if frame_end <= cursor:
                    raise RuntimeError("Satellite frame boundary did not advance")
                guard_time += frame_end-cursor
                cursor = frame_end
                continue
            boundary = min(float(self.times[index+1])+shift, data_end)
            next_block = candidate+1
            while next_block < len(self.blocked) and self.blocked[next_block,2] == 1 and not handover_enabled:
                next_block += 1
            if next_block < len(self.blocked):
                boundary = min(boundary, float(self.blocked[next_block,0])+shift)
            rates = self.series.get(link+"_bps")
            rate = min(ceiling_bps, float(rates[index])) if rates is not None else ceiling_bps
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("Unavailable radio interval must have a buffering window")
            available = max(0.0, boundary-cursor)
            if available == 0:
                cursor = np.nextafter(cursor, np.inf)
                continue
            duration = remaining/rate
            if duration <= available:
                # Finish explicitly even when the residual duration is below
                # one clock ULP. Advancing the clock without consuming this
                # fractional-bit remainder otherwise loops forever.
                transmitting += duration
                cursor += duration
                remaining = 0.0
            else:
                remaining = max(0.0, remaining-rate*available)
                transmitting += available
                cursor = boundary
        return cursor, transmitting, blocked_time, guard_time

    def finish(self, start, duration, handover_enabled=True):
        if start < self.times[0] or start > self.times[-1]:
            raise ValueError("Simulation exceeds orbital dataset timeline")
        cursor, remaining = start, duration
        for left, right, cause in self.blocked:
            if cause == 1 and not handover_enabled:
                continue
            if right <= cursor:
                continue
            available = max(0, left - cursor)
            if remaining <= available:
                break
            remaining -= available
            cursor = right
        finish = cursor + remaining
        if finish > self.times[-1]:
            raise ValueError(
                "Satellite service cannot finish within the orbital timeline"
            )
        return finish

    def delay(self, at, repeat=False):
        at = self.local_time(at, repeat)
        if not self.times[0] <= at <= self.times[-1]:
            raise ValueError("Simulation exceeds orbital dataset timeline")
        index = min(
            len(self.times) - 1, int(np.searchsorted(self.times, at, side="right")) - 1
        )
        # No interpolation across satellite changes or no-visibility intervals.
        value = float(self.propagation[index])
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                "No satellite service is available at transmission completion"
            )
        return value
