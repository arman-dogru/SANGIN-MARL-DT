"""Paper vertex-plus-edge metrics and a separate exploratory vertex variant."""

CONSISTENCY_SCHEMA = 2


def verify_ordering_quantities(pair):
    """Verify the three ordering indicators and three errors from raw priorities.

    Smaller PANAMA rho means higher application priority. The cached indicator
    compares cached peer AND cached self, never fresh physical self priority.
    """
    physical = int(pair["physical_peer_priority"] < pair["physical_self_priority"])
    twin = int(pair["digital_twin_peer_priority"] < pair["digital_twin_self_priority"])
    cached = int(pair["peer_priority"] < pair["self_priority"])
    expected = dict(current_order=physical, digital_twin_order=twin, cached_order=cached,
        uplink_ordering_error=int(physical != twin), downlink_ordering_error=int(twin != cached),
        e2e_ordering_error=int(physical != cached))
    for key, value in expected.items():
        if pair[key] != value:
            raise ValueError(f"Ordering equation mismatch: {key}")
    if pair["ordering_error"] != expected["e2e_ordering_error"]:
        raise ValueError("Ordering error alias disagrees with E2E error")
    if pair["e2e_ordering_error"] > pair["uplink_ordering_error"]+pair["downlink_ordering_error"]:
        raise ValueError("Ordering error decomposition violated")
    return True


def consistency_snapshot(positions, truth, states, now, radius=7, horizon=5, age_reference=1.0,
                         dt_states=None, action_views=None, cache_metadata=None, *, schema_version=CONSISTENCY_SCHEMA,
                         dt_references=None):
    if radius < 0 or horizon < 0 or age_reference <= 0:
        raise ValueError("Invalid FoV, prediction horizon, or age reference")
    if schema_version not in (1, CONSISTENCY_SCHEMA):
        raise ValueError("Unknown consistency schema")
    pairs = []

    def path(state):
        cells = [tuple(p) for p in state["path"]]
        if not cells:
            cells = [tuple(state["position"])]
        return (cells+[cells[-1]]*(horizon+1))[:horizon+1]

    for receiver, point in positions.items():
        receiver_dt = dt_references[receiver][1] if dt_references is not None else dt_states
        own = states[receiver][receiver]
        own_path = path(own.value)
        own_age = own.age(now)
        def visible(cell):
            return max(abs(cell[0]-point[0]), abs(cell[1]-point[1])) <= radius
        def intersections(first, second):
            vertex = [k for k in range(horizon+1) if first[k] == second[k] and visible(second[k])]
            edge = [k for k in range(horizon) if first[k] == second[k+1] and first[k+1] == second[k]
                    and first[k] != first[k+1] and visible(second[k]) and visible(second[k+1])]
            return vertex, edge
        for peer, cached in states[receiver].items():
            if peer == receiver:
                continue
            peer_path = path(cached.value)
            vertex, edge = intersections(own_path, peer_path)
            current = int(truth[peer]["priority"] < truth[receiver]["priority"])
            digital_twin = (int(receiver_dt[peer].value["priority"] < receiver_dt[receiver].value["priority"])
                            if receiver_dt is not None else current)
            perceived = int(cached.value["priority"] < own.value["priority"])
            error = abs(current-perceived)
            uplink, downlink = abs(digital_twin-current), abs(perceived-digital_twin)
            relevant = int(bool(vertex or edge))
            age = max(own_age, cached.age(now))
            physical_known = all("path" in truth[k] and "position" in truth[k] for k in (receiver, peer))
            physical_self = path(truth[receiver]) if physical_known else None
            physical_peer = path(truth[peer]) if physical_known else None
            pv, pe = intersections(physical_self, physical_peer) if physical_known else ([], [])
            physical_relevant = int(bool(pv or pe)) if physical_known else None
            view = (action_views or {}).get(receiver)
            action_known = view is not None and all(k in view["priorities"] for k in (receiver, peer))
            action_order = int(view["priorities"][peer] < view["priorities"][receiver]) if action_known else None
            dt_self = path(view["states"][receiver]) if action_known else None
            dt_peer = path(view["states"][peer]) if action_known else None
            dv, de = intersections(dt_self, dt_peer) if action_known else ([], [])
            metadata = (cache_metadata or {}).get(receiver, {})
            self_source = min(own.source_timestamps.values()) if own.source_timestamps else own.timestamp
            peer_source = min(cached.source_timestamps.values()) if cached.source_timestamps else cached.timestamp
            pair = dict(receiver=receiver, peer=peer, current_order=current,
                cached_order=perceived, ordering_error=error, relevant=relevant,
                digital_twin_order=digital_twin, uplink_ordering_error=uplink,
                downlink_ordering_error=downlink, e2e_ordering_error=error,
                fov_uplink_error=relevant*uplink, fov_downlink_error=relevant*downlink,
                fov_error=relevant*error, aoi_weighted_error=relevant*error*age/age_reference,
                self_age_seconds=own_age, peer_age_seconds=cached.age(now),
                self_source_time=self_source, peer_source_time=peer_source,
                self_priority=own.value["priority"], peer_priority=cached.value["priority"],
                self_completed=own.value["completed"], peer_completed=cached.value["completed"],
                vertex_conflict_steps=vertex, edge_conflict_steps=edge,
                within_square_fov=visible(positions[peer]), physical_conflict_known=physical_known,
                physical_vertex_conflict_steps=pv, physical_edge_conflict_steps=pe,
                physical_relevant=physical_relevant,
                false_positive_conflict=int(relevant and not physical_relevant) if physical_known else None,
                missed_physical_conflict=int(physical_relevant and not relevant) if physical_known else None,
                action_order_known=action_known, action_dt_order=action_order,
                action_dt_ordering_error=abs(current-action_order) if action_known else None,
                action_cache_ordering_error=abs(perceived-action_order) if action_known else None,
                action_dt_vertex_conflict_steps=dv, action_dt_edge_conflict_steps=de,
                physical_self_path=physical_self, physical_peer_path=physical_peer,
                cached_self_path=own_path, cached_peer_path=peer_path,
                action_dt_self_path=dt_self, action_dt_peer_path=dt_peer,
                physical_self_priority=truth[receiver]["priority"], physical_peer_priority=truth[peer]["priority"],
                action_dt_self_priority=view["priorities"][receiver] if action_known else None,
                action_dt_peer_priority=view["priorities"][peer] if action_known else None,
                self_generation_time=own.timestamp, peer_generation_time=cached.timestamp,
                self_arrival_time=metadata.get(receiver, {}).get("arrived_at"),
                peer_arrival_time=metadata.get(peer, {}).get("arrived_at"),
                self_version=own.version, peer_version=cached.version,
                self_pre_age_seconds=own.timestamp-self_source, self_post_age_seconds=now-own.timestamp,
                peer_pre_age_seconds=cached.timestamp-peer_source, peer_post_age_seconds=now-cached.timestamp)
            if schema_version >= 2:
                twin_self = receiver_dt[receiver].value if receiver_dt is not None else truth[receiver]
                twin_peer = receiver_dt[peer].value if receiver_dt is not None else truth[peer]
                vertex_relevant = int(bool(vertex))
                # Compare predicted (step, cell) events, not just existence of
                # any vertex conflict somewhere in the horizon.
                missed_steps = [k for k in pv if k not in vertex or physical_self[k] != own_path[k]]
                false_steps = [k for k in vertex if k not in pv or physical_self[k] != own_path[k]] if physical_known else None
                def twin_time(robot):
                    if receiver_dt is None:
                        return None
                    cache = receiver_dt[robot]
                    return dict(source_time=min(cache.source_timestamps.values()) if cache.source_timestamps else cache.timestamp,
                        generation_time=cache.timestamp, version=cache.version)
                def raw_path(state):
                    return [tuple(cell) for cell in state["path"]] if "path" in state else None
                pair.update(evaluated_at=now, vertex_relevant=vertex_relevant,
                    vertex_e2e_error=vertex_relevant*error,
                    vertex_aoi_weighted_error=vertex_relevant*error*age/age_reference,
                    physical_vertex_relevant=int(bool(pv)) if physical_known else None,
                    missed_vertex_conflict=int(bool(pv) and not vertex_relevant) if physical_known else None,
                    false_positive_vertex_conflict=int(vertex_relevant and not pv) if physical_known else None,
                    missed_vertex_conflict_steps=missed_steps if physical_known else None,
                    false_positive_vertex_conflict_steps=false_steps,
                    digital_twin_available=receiver_dt is not None,
                    digital_twin_self_priority=twin_self["priority"], digital_twin_peer_priority=twin_peer["priority"],
                    digital_twin_self_path=path(twin_self) if "path" in twin_self else None,
                    digital_twin_peer_path=path(twin_peer) if "path" in twin_peer else None,
                    digital_twin_self_timestamps=twin_time(receiver), digital_twin_peer_timestamps=twin_time(peer),
                    raw_intentions=dict(physical_self=raw_path(truth[receiver]), physical_peer=raw_path(truth[peer]),
                        digital_twin_self=raw_path(twin_self), digital_twin_peer=raw_path(twin_peer),
                        cached_self=raw_path(own.value), cached_peer=raw_path(cached.value)))
                pair["ordering_equations_verified"] = verify_ordering_quantities(pair)
            if dt_references is not None:
                pair["digital_twin_location"] = dt_references[receiver][0]
            pairs.append(pair)
    return summarize_pairs(pairs, truth, schema_version, dt_references)


def summarize_pairs(pairs, truth, schema_version=CONSISTENCY_SCHEMA, dt_references=None):
    """Aggregate directed pair records (factored out unchanged for receiver filtering)."""
    relevant = sum(p["relevant"] for p in pairs)
    errors = sum(p["fov_error"] for p in pairs)
    uplink = sum(p["fov_uplink_error"] for p in pairs)
    downlink = sum(p["fov_downlink_error"] for p in pairs)
    result = dict(pairs=pairs, directed_pairs=len(pairs),
        ordering_errors=sum(p["ordering_error"] for p in pairs),
        relevant_pairs=relevant, fov_errors=errors,
        uplink_errors=sum(p["uplink_ordering_error"] for p in pairs),
        downlink_errors=sum(p["downlink_ordering_error"] for p in pairs),
        fov_uplink_errors=uplink, fov_downlink_errors=downlink,
        zero_relevant_pair_case=relevant == 0,
        physical_relevant_pairs=sum(p["physical_relevant"] or 0 for p in pairs),
        missed_physical_conflicts=sum(p["missed_physical_conflict"] or 0 for p in pairs),
        false_positive_conflicts=sum(p["false_positive_conflict"] or 0 for p in pairs),
        vertex_relevant_pairs=sum(bool(p["vertex_conflict_steps"]) for p in pairs),
        edge_relevant_pairs=sum(bool(p["edge_conflict_steps"]) for p in pairs),
        unknown_physical_pairs=sum(not p["physical_conflict_known"] for p in pairs),
        action_order_known_pairs=sum(p["action_order_known"] for p in pairs),
        action_dt_ordering_errors=sum(p["action_dt_ordering_error"] or 0 for p in pairs),
        mean_peer_source_age_seconds=sum(p["peer_age_seconds"] for p in pairs)/max(1, len(pairs)),
        max_peer_source_age_seconds=max((p["peer_age_seconds"] for p in pairs), default=0.0),
        xi_fov_uplink=uplink/max(1, relevant), xi_fov_downlink=downlink/max(1, relevant),
        xi_fov=errors/max(1, relevant),
        xi_aoi=sum(p["aoi_weighted_error"] for p in pairs)/max(1, relevant))
    if schema_version >= 2:
        vertices = result["vertex_relevant_pairs"]
        vertex_errors = sum(p["vertex_e2e_error"] for p in pairs)
        vertex_aoi = sum(p["vertex_aoi_weighted_error"] for p in pairs)
        result.update(consistency_schema=CONSISTENCY_SCHEMA,
            ordering_pairs_verified=sum(p["ordering_equations_verified"] for p in pairs),
            vertex_e2e_errors=vertex_errors, vertex_aoi_error_sum=vertex_aoi,
            xi_vertex_e2e=vertex_errors/max(1, vertices), xi_vertex_aoi=vertex_aoi/max(1, vertices),
            zero_vertex_relevant_pair_case=vertices == 0,
            physical_vertex_relevant_pairs=sum(p["physical_vertex_relevant"] or 0 for p in pairs),
            missed_vertex_conflicts=sum(p["missed_vertex_conflict"] or 0 for p in pairs),
            false_positive_vertex_conflicts=sum(p["false_positive_vertex_conflict"] or 0 for p in pairs),
            missed_vertex_conflict_step_count=sum(len(p["missed_vertex_conflict_steps"] or []) for p in pairs),
            false_positive_vertex_conflict_step_count=sum(len(p["false_positive_vertex_conflict_steps"] or []) for p in pairs))
    if dt_references is not None:
        result['physical_relevant_ordering_errors']=sum((p['physical_relevant'] or 0)*p['ordering_error'] for p in pairs)
        active_pairs=[p for p in pairs if not truth[p['receiver']]['completed'] and not truth[p['peer']]['completed']]
        result['active_peer_aoi_sum']=sum(p['peer_age_seconds'] for p in active_pairs)
        result['active_peer_aoi_samples']=len(active_pairs)
    return result
