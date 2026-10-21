"""Directed square-FoV ordering errors, Equations 28--36 of the paper."""


def consistency_snapshot(positions, truth, states, now, radius=7, horizon=5, age_reference=1.0,
                         dt_states=None, action_views=None, cache_metadata=None):
    if radius < 0 or horizon < 0 or age_reference <= 0:
        raise ValueError("Invalid FoV, prediction horizon, or age reference")
    pairs = []

    def path(state):
        cells = [tuple(p) for p in state["path"]]
        if not cells:
            cells = [tuple(state["position"])]
        return (cells+[cells[-1]]*(horizon+1))[:horizon+1]

    for receiver, point in positions.items():
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
            digital_twin = (int(dt_states[peer].value["priority"] < dt_states[receiver].value["priority"])
                            if dt_states is not None else current)
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
            pairs.append(pair)
    relevant = sum(p["relevant"] for p in pairs)
    errors = sum(p["fov_error"] for p in pairs)
    uplink = sum(p["fov_uplink_error"] for p in pairs)
    downlink = sum(p["fov_downlink_error"] for p in pairs)
    return dict(pairs=pairs, directed_pairs=len(pairs),
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
