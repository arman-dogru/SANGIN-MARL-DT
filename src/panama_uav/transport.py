"""Deterministic virtual-time transport. Nothing in this module sleeps."""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import TransportConfig
from .leo import OrbitTimeline
from .event_calendar import EventCalendar


def encode(payload):
    # Disable reference memoization: identical value tuples have identical encoded sizes
    # before and after checkpoint loading, even if NumPy dtype aliases differ.
    stream = io.BytesIO()
    pickler = pickle.Pickler(stream, protocol=5)
    pickler.fast = True
    pickler.dump(payload)
    return stream.getvalue()


@dataclass(frozen=True)
class Message:
    id: int
    agent: int
    kind: str
    generated_at: float
    version: int
    wire: bytes
    route: tuple[str, ...]

    @property
    def size_bytes(self):
        return len(self.wire)

    @property
    def payload(self):
        return pickle.loads(self.wire)


@dataclass
class Queued:
    message: Message
    hop: int
    entered_at: float
    ready_at: float
    remaining_bytes: int | None = None
    latest_arrival: float = 0.0
    transmitted_bytes: int = 0
    transmission_seconds: float = 0.0
    interruption_seconds: float = 0.0
    guard_seconds: float = 0.0
    queue_seconds: float = 0.0
    chunks: int = 0
    first_start: float | None = None


@dataclass
class Link:
    capacity_bps: float
    queues: dict = field(default_factory=lambda: defaultdict(deque))
    busy: bool = False
    last_agent: int = -1


@dataclass(frozen=True)
class DelayTrace:
    timestamps: np.ndarray
    delays: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["timestamps"], data["delays"], json.loads(str(data["metadata"]))
            )

    def delay(self, time):
        if time < self.timestamps[0] or time > self.timestamps[-1]:
            raise ValueError(
                "Simulation exceeds imported trace segment; select a longer segment"
            )
        index = max(0, int(np.searchsorted(self.timestamps, time, side="right")) - 1)
        return float(self.delays[index])


def import_trace(
    source, output, direction, unit="ns", start=None, end=None, origin_seconds=None
):
    """Import a three-column SATPIPE/Starlink OWD file, without treating sequence gaps as loss."""
    source = Path(source)
    if direction not in ("uplink", "downlink"):
        raise ValueError("Direction is required")
    scales = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
    if unit not in scales:
        raise ValueError("Explicit supported units required")
    # Read strings before conversion to avoid rounding integer nanosecond timestamps.
    raw = np.loadtxt(
        source,
        dtype=str,
        delimiter="," if "," in source.read_text().splitlines()[0] else None,
        ndmin=2,
    )
    if raw.shape[1] != 3:
        raise ValueError("Expected sequence, transmit timestamp, one-way delay")
    column_names = None
    if (
        raw[0, 0] == "seq_nr"
        and raw[0, 1] in ("ul_t0", "dl_t0")
        and raw[0, 2] == "owd_t0_t1"
    ):
        column_names = raw[0].tolist()
        raw = raw[1:]
    if not len(raw):
        raise ValueError("Empty trace file")
    timestamps = np.array([float(value) for value in raw[:, 1]]) * scales[unit]
    delays = raw[:, 2].astype(float) * scales[unit]
    if (
        not np.all(np.isfinite(timestamps))
        or not np.all(np.isfinite(delays))
        or np.any(delays < 0)
    ):
        raise ValueError("Invalid trace values")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Trace timestamps must be ordered")
    mask = np.ones(len(raw), bool)
    if start is not None:
        mask &= timestamps >= start
    if end is not None:
        mask &= timestamps <= end
    if not mask.any():
        raise ValueError("Empty trace segment")
    chosen = timestamps[mask]
    origin = float(chosen[0]) if origin_seconds is None else float(origin_seconds)
    if origin > chosen[0]:
        raise ValueError("Trace origin must not be later than its first sample")
    metadata = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "direction": direction,
        "source_unit": unit,
        "stored_unit": "seconds",
        "source_first_timestamp": str(raw[mask, 1][0]),
        "source_last_timestamp": str(raw[mask, 1][-1]),
        "segment_start_seconds": origin,
        "segment_end_seconds": float(chosen[-1]),
        "rows": int(mask.sum()),
        "column_names": column_names,
        "time_origin_seconds": origin,
        "missing_sequences_are_loss": False,
        "replacement_scope": "entire satellite backhaul hop; access remains modeled",
        "handover_identification": "not inferred from delay spikes",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        timestamps=chosen - origin,
        delays=delays[mask],
        sequence=raw[mask, 0],
        source_timestamps=raw[mask, 1],
        metadata=json.dumps(metadata),
    )
    return metadata


class Transport:
    def __init__(self, config: TransportConfig):
        self.config = config
        self.now = 0.0
        self.sequence = 0
        self.message_sequence = 0
        self.next_handover_index = 1
        self.satellite_assignment = 0
        self.calendar = EventCalendar()
        self.links = {
            name: Link(capacity * 1e6)
            for name, capacity in (
                ("access_up", config.access_uplink_mbps),
                ("access_down", config.access_downlink_mbps),
                ("satellite_up", config.satellite_uplink_mbps),
                ("satellite_down", config.satellite_downlink_mbps),
            )
        }
        self.windows = {}
        self.flow_ready_at = {}
        self.log = []
        self.counters = defaultdict(int)
        self.traces = {}
        self.orbit = (
            OrbitTimeline.load(config.leo_timeline)
            if config.leo_timeline and config.profile == "physical" else None
        )
        self.next_orbit_event = 0
        self.orbit_logged_until = -1e-9
        if self.orbit is not None and self.orbit.metadata.get("stress_handover_seconds", 0) and not config.handover_enabled:
            raise ValueError("Stress timelines require interruptions enabled; use the native timeline for handover-off")
        for name, path, direction in (
            ("satellite_up", config.uplink_trace, "uplink"),
            ("satellite_down", config.downlink_trace, "downlink"),
        ):
            if path:
                trace = DelayTrace.load(path)
                if trace.metadata["direction"] != direction:
                    raise ValueError("Trace direction mismatch")
                self.traces[name] = trace

    def schedule_episode(self, seed, agents, cap=250):
        rng = np.random.default_rng(seed)
        self.windows = {}
        if not self.config.contention_enabled:
            return
        for agent in range(agents):
            count = int(
                rng.integers(self.config.windows_min, self.config.windows_max + 1)
            )
            starts = sorted(
                rng.choice(np.arange(1, cap), count, replace=False).tolist()
            )
            self.windows[agent] = [
                (
                    self.now + s * self.config.seconds_per_step,
                    self.now
                    + (s + self.config.reservation_steps)
                    * self.config.seconds_per_step,
                )
                for s in starts
            ]
            for start, end in self.windows[agent]:
                self.log.append(
                    {
                        "event": "scheduled_reservation",
                        "time": self.now,
                        "agent": agent,
                        "start": start,
                        "end": end,
                        "seed": seed,
                    }
                )

    def _event(self, at, kind, value):
        self.sequence += 1
        self.calendar.schedule(float(at), kind, value)

    @property
    def events(self):
        return self.calendar.records()

    def send(
        self,
        agent,
        kind,
        payload=None,
        *,
        route=(),
        generated_at=None,
        version=0,
        wire=None,
        delayed=True,
    ):
        if (
            kind in ("model", "model_relay")
            and self.config.profile == "physical"
            and any(name in self.traces for name in route)
        ):
            raise ValueError(
                "UDP packet-delay traces do not calibrate bulk model snapshots; use the orbital capacity model"
            )
        self.message_sequence += 1
        body = encode(payload) if wire is None else wire
        # The envelope is transmitted, too; the receiver decodes only complete messages.
        envelope = encode(
            {
                "agent": agent,
                "kind": kind,
                "timestamp": self.now if generated_at is None else generated_at,
                "version": version,
            }
        )
        packet = Message(
            self.message_sequence,
            agent,
            kind,
            self.now if generated_at is None else generated_at,
            version,
            body,
            tuple(route),
        )
        self.counters["messages_generated"] += 1
        self.counters["bytes_generated"] += len(body) + len(envelope)
        self.log.append(
            {
                "event": "generated",
                "time": self.now,
                "id": packet.id,
                "agent": agent,
                "kind": kind,
                "bytes": len(body) + len(envelope),
                "version": version,
                "source_time": packet.generated_at,
            }
        )
        release = self.now
        if (
            delayed
            and self.config.profile != "perfect"
            and self.config.contention_enabled
        ):
            for start, end in self.windows.get(agent, []):
                if start <= self.now < end:
                    release += self.config.delay_steps * self.config.seconds_per_step
                    self.counters["exposed_messages"] += 1
                    self.log.append(
                        {
                            "event": "reservation_exposure",
                            "time": self.now,
                            "agent": agent,
                            "id": packet.id,
                            "start": start,
                            "end": end,
                            "release": release,
                        }
                    )
                    break
        if delayed and self.config.profile != "perfect":
            # FIFO buffering prevents newer updates from bypassing a reserved flow.
            # This is essential: a 20-step hold must create 20-step information age,
            # rather than merely delivering an obsolete packet after fresh ones.
            key = (agent, kind)
            release = max(release, self.flow_ready_at.get(key, self.now))
            self.flow_ready_at[key] = release
            if release > self.now:
                self.log.append(
                    {
                        "event": "buffered",
                        "time": self.now,
                        "agent": agent,
                        "id": packet.id,
                        "ready_at": release,
                        "buffer_seconds": release - self.now,
                    }
                )
        # Keep envelope size as part of the actual service budget without copying large model bytes.
        self._event(release, "enqueue", (packet, 0, len(envelope)))
        return packet.id

    def _finish_transmission(self, start, duration, satellite):
        if satellite and self.orbit is not None:
            offset = self.config.backhaul_time_offset_seconds
            return (
                self.orbit.finish(
                    start + offset, duration, self.config.handover_enabled
                )
                - offset
            )
        if not satellite or not self.config.handover_enabled:
            return start + duration
        interval = self.config.handover_interval_seconds
        pause = self.config.handover_interruption_seconds
        remaining = duration
        time = start
        while remaining > 1e-14:
            index = math.floor(time / interval)
            window = index * interval
            if index > 0 and time < window + pause:
                time = window + pause
            next_window = (math.floor(time / interval) + 1) * interval
            available = next_window - time
            if remaining <= available:
                return time + remaining
            remaining -= available
            time = next_window + pause
        return time

    def _start(self, name):
        link = self.links[name]
        if link.busy:
            return
        eligible = sorted(
            a
            for a, queue in link.queues.items()
            if queue and queue[0][0].ready_at <= self.now + 1e-12
        )
        if not eligible:
            return
        agent = next((a for a in eligible if a > link.last_agent), eligible[0])
        queued, header = link.queues[agent].popleft()
        link.last_agent = agent
        link.busy = True
        satellite = name.startswith("satellite")
        if queued.remaining_bytes is None:
            queued.remaining_bytes = queued.message.size_bytes + header
        fragmented = self.config.packet_payload_bytes > 0
        served = min(queued.remaining_bytes, self.config.packet_payload_bytes) if fragmented else queued.remaining_bytes
        # Chunk-level scheduling aggregates ordinary 1500-byte packets.
        wire_bytes = served + (math.ceil(served/1500)*self.config.packet_overhead_bytes if fragmented else 0)
        duration = wire_bytes * 8 / link.capacity_bps
        guard = 0.0
        if satellite and self.orbit is not None and fragmented:
            offset = self.config.backhaul_time_offset_seconds
            finish, duration, interrupted, guard = self.orbit.service(
                self.now+offset, wire_bytes*8, name, link.capacity_bps,
                self.config.scheduler_frame_seconds, self.config.scheduler_guard_fraction,
                self.config.orbital_replay, self.config.handover_enabled,
            )
            finish -= offset
        else:
            finish = self._finish_transmission(self.now, duration, satellite)
            interrupted = finish-self.now-duration
        propagation = self.config.one_way_propagation_seconds if satellite else 0.0
        if satellite and self.orbit is not None:
            propagation = self.orbit.delay(
                finish + self.config.backhaul_time_offset_seconds,
                self.config.orbital_replay,
            )
        queued.transmitted_bytes += wire_bytes
        queued.transmission_seconds += duration
        queued.interruption_seconds += interrupted
        queued.guard_seconds += guard
        queued.queue_seconds += self.now-queued.ready_at
        queued.chunks += 1
        if queued.first_start is None:
            queued.first_start = self.now
        queued.remaining_bytes -= served
        if self.config.log_service_chunks or queued.remaining_bytes == 0:
            self.log.append(
            {
                "event": "service_chunk" if queued.remaining_bytes else "transmit",
                "time": queued.first_start if queued.remaining_bytes == 0 else self.now,
                "id": queued.message.id,
                "agent": agent,
                "link": name,
                "bytes": queued.transmitted_bytes if queued.remaining_bytes == 0 else wire_bytes,
                "payload_bytes": queued.message.size_bytes+header if queued.remaining_bytes == 0 else served,
                "chunks": queued.chunks,
                "kind": queued.message.kind,
                "queue_seconds": queued.queue_seconds,
                "transmission_seconds": queued.transmission_seconds,
                "handover_seconds": queued.interruption_seconds,
                "frame_guard_seconds": queued.guard_seconds,
                "propagation_seconds": propagation,
                "finish": finish,
            }
        )
        queued.ready_at = finish
        queued.latest_arrival = max(queued.latest_arrival, finish+propagation)
        self._event(finish, "free", (name, queued, header) if queued.remaining_bytes else name)
        if not queued.remaining_bytes:
            self._event(queued.latest_arrival, "enqueue", (queued.message, queued.hop+1, header))
        self.counters["hop_bytes_transmitted"] += wire_bytes

    def advance(self, until, receiver=None):
        if until < self.now - 1e-12:
            raise ValueError("Transport time cannot reverse")
        delivered = []
        if (
            self.config.profile == "physical"
            and self.orbit is not None
            and (self.config.handover_enabled or self.orbit.metadata.get("handover_study"))
        ):
            offset = self.config.backhaul_time_offset_seconds
            for event in self.orbit.events_between(self.orbit_logged_until+offset, until+offset,
                                                   self.config.orbital_replay):
                if self.orbit.metadata.get("handover_study"):
                    event = dict(event)
                    applied = event.get("service_category") != "satellite_change" or self.config.handover_enabled
                    event["interruption_applied"] = applied
                    if not applied:
                        event["end"] = event["start"]
                        event["control_messages"] = []
                self._event(
                    max(self.now, event["start"] - offset), "orbit_handover", event
                )
                for control in event.get("control_messages", []):
                    self._event(max(self.now,control["start"]-offset), "handshake_packet", control)
                self._event(event["end"]-offset, "orbit_ready", event)
            self.orbit_logged_until = until
        elif self.config.profile == "physical" and self.config.handover_enabled:
            while (
                self.next_handover_index * self.config.handover_interval_seconds
                <= until
            ):
                at = self.next_handover_index * self.config.handover_interval_seconds
                self._event(at, "handover_start", self.next_handover_index)
                self._event(
                    at + self.config.handover_interruption_seconds,
                    "handover_end",
                    self.next_handover_index,
                )
                self.next_handover_index += 1
        self._event(until, "clock", None)
        while self.calendar.peek() <= until + 1e-12:
            at, _, kind, value = self.calendar.pop()
            self.now = at
            if kind == "clock":
                continue
            if kind in ("handshake_packet", "orbit_ready"):
                self.log.append({"event": kind, "time": at, "details": value})
                continue
            if kind == "orbit_handover":
                self.satellite_assignment = value["to_satellite"]
                self.log.append(
                    {
                        "event": "orbit_handover",
                        "time": at,
                        "assignment": self.satellite_assignment,
                        "source_window": value,
                        "assumed_handshake": value.get("interruption_applied", True),
                    }
                )
                continue
            if kind in ("handover_start", "handover_end"):
                if kind == "handover_start":
                    self.satellite_assignment = value
                self.log.append(
                    {
                        "event": kind,
                        "time": at,
                        "assignment": value,
                        "affects_links": [
                            name
                            for name in ("satellite_up", "satellite_down")
                            if name not in self.traces
                        ],
                    }
                )
                continue
            if kind == "free":
                if isinstance(value, tuple):
                    name, queued, header = value
                    self.links[name].queues[queued.message.agent].appendleft((queued, header))
                else:
                    name = value
                self.links[name].busy = False
                self._start(name)
                continue
            packet, hop, header = value
            if hop == len(packet.route):
                self.counters["messages_delivered"] += 1
                self.counters["bytes_delivered"] += packet.size_bytes + header
                self.log.append(
                    {
                        "event": "delivered",
                        "time": at,
                        "id": packet.id,
                        "agent": packet.agent,
                        "kind": packet.kind,
                        "source_time": packet.generated_at,
                        "age_seconds": at - packet.generated_at,
                        "version": packet.version,
                        "bytes": packet.size_bytes + header,
                    }
                )
                delivered.append(packet)
                if receiver is not None:
                    receiver(packet)
                continue
            name = packet.route[hop]
            if name.startswith("process:"):
                self._event(
                    at + float(name.split(":")[1]), "enqueue", (packet, hop + 1, header)
                )
            elif self.config.profile != "physical":
                self._event(at, "enqueue", (packet, hop + 1, header))
            elif name in self.traces:
                delay = self.traces[name].delay(
                    at + self.config.backhaul_time_offset_seconds
                )
                self.log.append(
                    {
                        "event": "trace_delay",
                        "time": at,
                        "id": packet.id,
                        "link": name,
                        "delay_seconds": delay,
                    }
                )
                self._event(at + delay, "enqueue", (packet, hop + 1, header))
            else:
                self.links[name].queues[packet.agent].append(
                    (Queued(packet, hop, at, at), header)
                )
                self._start(name)
        self.now = float(until)
        return delivered

    def occupancy(self):
        result = {
            name: sum(
                q.remaining_bytes if q.remaining_bytes is not None else q.message.size_bytes + h
                for values in link.queues.values()
                for q, h in values
            )
            for name, link in self.links.items()
        }
        result["reservation_buffer"] = sum(
            packet.size_bytes + header
            for at, _, kind, value in self.events
            if kind == "enqueue"
            for packet, hop, header in [value]
            if hop == 0 and at > self.now
        )
        result["in_flight"] = sum(
            packet.size_bytes + header
            for at, _, kind, value in self.events
            if kind == "enqueue"
            for packet, hop, header in [value]
            if hop > 0
        )
        result["transmitting_remaining"] = sum(
            value[1].remaining_bytes
            for at, _, kind, value in self.events
            if kind == "free" and isinstance(value, tuple)
        )
        return result

    def handover_windows(self, start, end):
        if self.orbit is not None and self.orbit.metadata.get("handover_study"):
            offset = self.config.backhaul_time_offset_seconds
            return [[max(start, e["start"]-offset), min(end, e["end"]-offset)]
                    for e in self.orbit.events_between(start+offset-self.orbit.period, end+offset, self.config.orbital_replay)
                    if e["end"]-offset > start and e["start"]-offset < end
                    and (self.config.handover_enabled or e.get("service_category") != "satellite_change")]
        if self.config.profile != "physical" or not self.config.handover_enabled:
            return []
        if self.orbit is not None:
            offset = self.config.backhaul_time_offset_seconds
            events = self.orbit.events_between(start+offset-1, end+offset, self.config.orbital_replay)
            return [[max(start,e["start"]-offset),min(end,e["end"]-offset)]
                    for e in events if e["end"]-offset >= start]
        interval = self.config.handover_interval_seconds
        return [
            [i * interval, i * interval + self.config.handover_interruption_seconds]
            for i in range(
                max(1, math.floor(start / interval)), math.floor(end / interval) + 1
            )
            if i * interval + self.config.handover_interruption_seconds >= start
        ]
