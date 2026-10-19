"""Picklable SimPy event calendar; no generators or wall-clock waiting."""

from collections import deque

import simpy


class EventCalendar:
    def __init__(self):
        self.environment = simpy.Environment()
        self.pending = {}
        self.completed = deque()
        self.sequence = 0

    def schedule(self, at, kind, value):
        if at < self.environment.now - 1e-10:
            raise ValueError("Event cannot be scheduled in the past")
        self.sequence += 1
        token = self.sequence
        self._install(at, token, kind, value)

    def _install(self, at, token, kind, value):
        event = simpy.events.Timeout(
            self.environment, max(0.0, at - self.environment.now), token
        )
        event.callbacks.append(self._receive)
        self.pending[token] = (float(at), token, kind, value)

    def __getstate__(self):
        # SimPy Environment's bound factory aliases are not pickle-portable.
        # Persist only the complete deterministic calendar and reconstruct it.
        return {"now": self.environment.now, "sequence": self.sequence,
                "pending": self.records(), "completed": list(self.completed)}

    def __setstate__(self, state):
        self.environment = simpy.Environment(initial_time=state["now"])
        self.sequence = state["sequence"]
        self.pending = {}
        self.completed = deque(state["completed"])
        for at, token, kind, value in sorted(state["pending"], key=lambda row: row[1]):
            self._install(at, token, kind, value)

    def _receive(self, event):
        self.completed.append(self.pending.pop(event.value))

    def peek(self):
        return self.environment.peek()

    def pop(self):
        self.environment.step()
        return self.completed.popleft()

    def records(self):
        return list(self.pending.values())
