"""Read-only history of accepted replica arrivals for controlled stress tests."""
from collections import defaultdict
import math

from .remote_state import DTReplica, DTState


class ReplicaHistory:
    def __init__(self):
        self.replicas = {name: DTReplica(name) for name in ('drone', 'cloud')}
        self.arrivals = defaultdict(list)

    def observe(self, event, by_id):
        kind = event['event']
        if kind == 'dt_state_event' and (event.get('bootstrap') or event.get('ideal_reference')):
            location, state, ref = event['location'], event['state'], event['event_id']
            available = event['available_at']
        elif kind == 'dt_state_acceptance':
            original = by_id[event['state_event']]
            location, state, ref = original['location'], original['state'], original['event_id']
            available = event['time']
        elif kind == 'replica_sync_arrived':
            location, state, ref = event['location'], event['state'], event['state_event']
            available = event['arrived_at']
        else:
            return
        state = DTState(**state)
        accepted = self.replicas[location].accept(state, available, ref)
        if 'accepted' in event and accepted != event['accepted']:
            raise ValueError('Controlled history acceptance mismatch')
        if accepted:
            self.arrivals[location, state.robot_id].append((float(available), ref, state))

    def view(self, location, robot, now, steps, interval=1.0):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps not in range(6):
            raise ValueError('Controlled peer delay must be an integer from 0 to 5')
        if not math.isfinite(now) or now < 0 or interval != 1.0:
            raise ValueError('Controlled study uses finite time and 1-second steps')
        current = self.replicas[location]
        if steps == 0:
            return current
        result = DTReplica(location)
        cutoff = now - steps * interval
        for peer in current.states:
            entries = self.arrivals[location, peer]
            if peer == robot:
                selected = entries[-1]
            else:
                eligible = [entry for entry in entries if entry[0] <= cutoff + 1e-12]
                # The original t=0 briefing is explicitly retained during warm-up.
                selected = eligible[-1] if eligible else entries[0]
                if not eligible and selected[0] != 0:
                    raise ValueError('Missing fresh initial briefing')
            available, ref, state = selected
            result.accept(state, now, ref)
        return result
