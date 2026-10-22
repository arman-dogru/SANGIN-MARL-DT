"""Controlled inference-only peer buffering; the production kernels are reused."""
import hashlib
import json
import numpy as np

from .consistency import consistency_snapshot
from .remote import RemoteDTCommunication
from .remote_state import DTReplica, observation_hash, remote_observation
from .staleness_state import ReplicaHistory


class ControlledRemoteDT(RemoteDTCommunication):
    delay_steps = 0

    def __init__(self, *args, **kwargs):
        self.peer_history = ReplicaHistory()
        self.stress_checksum = '0' * 64
        self.stress_settings = dict(schema=1, peer_delay_steps=self.delay_steps,
            scope='inference_peer_read_only', step_seconds=1.0,
            bootstrap='retain t=0 briefing until eligible history exists',
            label='controlled stress experiment; not nominal network delay')
        super().__init__(*args, **kwargs)

    def _record(self, event_name, **values):
        event = super()._record(event_name, **values)
        self.peer_history.observe(event, self.events_by_id)
        return event

    def _inference_replica(self, location, report):
        # At zero delay use the actual object, including original acceptance order.
        if self.delay_steps == 0:
            return self.replicas[location]
        return self.peer_history.view(location, report.robot_id, self.transport.now, self.delay_steps)

    def _after_inference(self, report, own, location, observation, action, event):
        infer = self.shadow_inference
        if infer is None:
            raise ValueError('Controlled staleness requires frozen shadow inference')
        args = (self.project.map.fov_radius, self.project.map.astar_waypoints,
                self.project.map.communicated_path_steps, max(self.grid.shape))
        actual = self._inference_replica(location, report)
        network = self.replicas[location]
        fresh = DTReplica(location)
        fresh_states = {}
        for agent in self._agents:
            sensed = self._sense(agent, self._agents, report.cycle, self.transport.now)
            state = self._state_for_report(sensed, self.transport.now, report.cycle)
            fresh_states[agent.id] = state.payload()
            fresh.accept(state, self.transport.now, f'stress-oracle:{report.cycle}:{agent.id}')
        network_obs, _, _, _ = remote_observation(report, own, network, *args)
        fresh_obs, _, _, _ = remote_observation(report, own, fresh, *args)
        # Same own sensing/vector/occupancy, same model and compute instant.
        for reference in (network_obs, fresh_obs):
            if not (np.array_equal(reference['spatial'][:2], observation['spatial'][:2])
                    and np.array_equal(reference['vector'], observation['vector'])):
                raise AssertionError('Staleness intervention changed self sensing')
        network_action = int(infer(location, report.robot_id, network_obs))
        fresh_action = int(infer(location, report.robot_id, fresh_obs))
        stores = {peer: {peer: cache} for peer, cache in fresh.states.items()}
        stores[report.robot_id] = dict(actual.states)
        # Own is the same locally computed state used by the real observation.
        own_replica = DTReplica(location)
        own_replica.accept(own, self.transport.now, event['dt_versions'][str(report.robot_id)]['state_event'])
        stores[report.robot_id][report.robot_id] = own_replica.states[report.robot_id]
        positions = {peer: state['position'] for peer,state in fresh_states.items()}
        positions[report.robot_id] = report.position
        snapshot = consistency_snapshot(positions, fresh_states, stores,
            self.transport.now, args[0], args[2], self.parameters.age_reference_seconds,
            network.states, {report.robot_id: dict(priorities={int(k): v for k, v in event['dt_priorities'].items()},
                states={peer: cache.value for peer, cache in stores[report.robot_id].items()})})
        diagnostic = dict(event='controlled_peer_shadow', time=self.transport.now,
            previous_checksum=self.stress_checksum, inference_event=event['event_id'],
            job_id=event['job_id'], robot_id=report.robot_id, location=location,
            delay_steps=self.delay_steps, actual_action=action,
            network_peer_action=network_action, fresh_peer_action=fresh_action,
            injected_delay_disagreement=action != network_action,
            stale_vs_fresh_disagreement=action != fresh_action,
            network_observation={k: v.tolist() for k, v in network_obs.items()},
            fresh_observation={k: v.tolist() for k, v in fresh_obs.items()},
            network_hash=observation_hash(network_obs), fresh_hash=observation_hash(fresh_obs),
            fresh_states=fresh_states, pairs=snapshot['pairs'],
            summary={k: v for k, v in snapshot.items() if k != 'pairs'}, evaluation_only=True)
        # A separate diagnostic chain preserves causal event IDs and wire sizes,
        # including exact zero-delay equivalence with the original controller.
        diagnostic = json.loads(json.dumps(diagnostic, allow_nan=False))
        canonical = json.dumps(diagnostic, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self.stress_checksum = hashlib.sha256(canonical).hexdigest()
        diagnostic['checksum'] = self.stress_checksum
        self.oracle_events.append(diagnostic)


def controlled_class(steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps not in range(6):
        raise ValueError('Controlled peer delay must be 0--5 steps')
    return type(f'ControlledRemoteDT{steps}', (ControlledRemoteDT,), dict(delay_steps=steps))
