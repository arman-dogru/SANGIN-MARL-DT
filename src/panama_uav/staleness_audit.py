"""Replay both causal inputs and independently chained stress diagnostics."""
from dataclasses import replace
import numpy as np

from .consistency import consistency_snapshot
from .environment import astar_path
from .remote_audit import digest, replay_trace
from .remote_state import DTReplica, DTState, PhysicalReport, observation_hash, remote_observation
from .staleness_state import ReplicaHistory


def audit_stress(trace, inference=None):
    result = replay_trace(trace, inference)
    settings = trace['remote_settings']
    stress = settings['controlled_staleness']
    if stress['scope'] != 'inference_peer_read_only' or stress['step_seconds'] != 1:
        raise ValueError('Unknown controlled intervention')
    steps = stress['peer_delay_steps']
    history = ReplicaHistory()
    by_id, reports, physical, positions = {}, {}, {}, {}
    diagnostics = [e for e in trace['oracle_events'] if e['event'] == 'controlled_peer_shadow']
    shadows = {e['inference_event']: e for e in diagnostics}
    if len(shadows) != len(diagnostics):
        raise ValueError('Duplicate peer shadow')
    previous = '0'*64
    for shadow in diagnostics:
        if shadow['previous_checksum'] != previous or digest({k:v for k,v in shadow.items() if k != 'checksum'}) != shadow['checksum']:
            raise ValueError('Peer shadow checksum mismatch')
        previous = shadow['checksum']
    rebuilt = 0
    args = settings['radius'], settings['waypoints'], settings['horizon'], settings['map_extent']
    grid = np.asarray(trace['grid'])
    for event in trace['remote_events']:
        by_id[event['event_id']] = event
        kind, now = event['event'], event['time']
        if kind == 'robot_state_event':
            report = PhysicalReport(**event['report'])
            reports[report.report_id] = report
            physical[report.robot_id] = event['physical']
            positions[report.robot_id] = tuple(report.position)
        elif kind == 'motion_event':
            physical = {int(k):v for k,v in event['physical'].items()}
            positions = {int(k):tuple(v) for k,v in event['after'].items()}
        elif kind == 'remote_inference':
            shadow = shadows.pop(event['event_id'], None)
            if shadow is None or shadow['time'] != now or shadow['delay_steps'] != steps:
                raise ValueError('Missing or mistimed controlled shadow')
            robot, location = event['robot_id'], event['location']
            own_ref = event['dt_versions'][str(robot)]['state_event']
            state = DTState(**by_id[own_ref]['state'])
            report = reports[state.report_id]
            path = astar_path(grid, tuple(report.position), tuple(report.goal))
            own = replace(state, path=path or [tuple(report.position)])
            actual = history.view(location, robot, now, steps)
            network = history.replicas[location]
            fresh = DTReplica(location)
            truth = {}
            for peer, values in physical.items():
                peer_path = astar_path(grid, positions[peer], tuple(values['goal']))
                priority = (grid.size if values['completed'] else
                    (len(peer_path)-1 if peer_path else grid.size-1)) + peer*1e-6
                fresh_state = DTState(peer, positions[peer], tuple(values['goal']), peer_path or [positions[peer]],
                    priority, values['completed'], now, report.cycle, {peer:now}, f'report:{report.cycle}:{peer}')
                fresh.accept(fresh_state, now, f'stress-oracle:{report.cycle}:{peer}')
                truth[peer] = fresh_state.payload()
            if digest(truth) != digest(shadow['fresh_states']):
                raise ValueError('Fresh peer oracle is not physical truth at inference')
            actual_obs = remote_observation(report, own, actual, *args)[0]
            for prefix, replica in (('network', network), ('fresh', fresh)):
                obs = remote_observation(report, own, replica, *args)[0]
                if (observation_hash(obs) != shadow[prefix+'_hash'] or
                    observation_hash(shadow[prefix+'_observation']) != shadow[prefix+'_hash']):
                    raise ValueError('Peer shadow observation mismatch')
                if not (np.array_equal(obs['spatial'][:2], actual_obs['spatial'][:2]) and
                        np.array_equal(obs['vector'], actual_obs['vector'])):
                    raise ValueError('Peer intervention modified self sensing')
                if inference and int(inference(location, robot, obs)) != shadow[prefix+'_peer_action']:
                    raise ValueError('Peer shadow model action mismatch')
            if (shadow['actual_action'] != event['action'] or
                shadow['injected_delay_disagreement'] != (event['action'] != shadow['network_peer_action']) or
                shadow['stale_vs_fresh_disagreement'] != (event['action'] != shadow['fresh_peer_action'])):
                raise ValueError('Peer shadow disagreement mismatch')
            stores = {peer:{peer:cache} for peer,cache in fresh.states.items()}
            stores[robot] = dict(actual.states)
            own_replica = DTReplica(location)
            own_replica.accept(own, now, own_ref)
            stores[robot][robot] = own_replica.states[robot]
            observation_positions = dict(positions)
            observation_positions[robot] = report.position
            snapshot = consistency_snapshot(observation_positions, truth, stores, now,
                args[0], args[2], settings['age_reference'], network.states,
                {robot:dict(priorities={int(k):v for k,v in event['dt_priorities'].items()},
                    states={peer:cache.value for peer,cache in stores[robot].items()})})
            if (digest(snapshot['pairs']) != digest(shadow['pairs']) or
                digest({k:v for k,v in snapshot.items() if k != 'pairs'}) != digest(shadow['summary'])):
                raise ValueError('Inference FoV/AoI mismatch')
            rebuilt += 1
        history.observe(event, by_id)
    if shadows or rebuilt != result['rebuilt_inferences']:
        raise ValueError('Controlled shadow coverage mismatch')
    active = {(e['cycle'],e['robot_id']) for e in trace['remote_events']
              if e['event']=='decision_event' and not e['completed_local_control']}
    cutoffs = [e for e in trace['oracle_events'] if e['event']=='decision_action_shadow']
    if len(cutoffs) != len(active) or {(e['cycle'],e['robot_id']) for e in cutoffs} != active:
        raise ValueError('Cutoff oracle coverage mismatch')
    for shadow in cutoffs:
        if observation_hash(shadow['observation']) != shadow['observation_hash']:
            raise ValueError('Cutoff oracle observation hash mismatch')
        if inference and int(inference('cloud',shadow['robot_id'],shadow['observation'])) != shadow['oracle_action']:
            raise ValueError('Cutoff oracle model action mismatch')
    return dict(status='passed', causal_events=result['events'],
        actual_actions_verified=result['model_actions_verified'], peer_shadows_rebuilt=rebuilt,
        peer_shadow_actions_verified=2*rebuilt if inference else 0,
        cutoff_oracles_verified=len(cutoffs) if inference else 0,
        ordering_pairs_verified=result['ordering_pairs_verified'], diagnostic_checksum=previous)
