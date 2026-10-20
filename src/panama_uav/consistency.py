"""Directed square-FoV ordering errors, Equations 28--36 of the paper."""


def consistency_snapshot(positions, truth, states, now, radius=7, horizon=5, age_reference=1.0):
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
        for peer, cached in states[receiver].items():
            if peer == receiver:
                continue
            peer_path = path(cached.value)
            vertex = [k for k in range(horizon+1)
                      if own_path[k] == peer_path[k] and visible(peer_path[k])]
            edge = [k for k in range(horizon)
                    if own_path[k] == peer_path[k+1] and own_path[k+1] == peer_path[k]
                    and visible(peer_path[k])]
            current = int(truth[peer]["priority"] < truth[receiver]["priority"])
            perceived = int(cached.value["priority"] < own.value["priority"])
            error = abs(current-perceived)
            relevant = int(bool(vertex or edge))
            age = max(own_age, cached.age(now))
            pairs.append(dict(receiver=receiver, peer=peer, current_order=current,
                cached_order=perceived, ordering_error=error, relevant=relevant,
                fov_error=relevant*error, aoi_weighted_error=relevant*error*age/age_reference,
                self_age_seconds=own_age, peer_age_seconds=cached.age(now),
                self_source_time=min(own.source_timestamps.values()) if own.source_timestamps else own.timestamp,
                peer_source_time=min(cached.source_timestamps.values()) if cached.source_timestamps else cached.timestamp,
                self_priority=own.value["priority"], peer_priority=cached.value["priority"],
                self_completed=own.value["completed"], peer_completed=cached.value["completed"],
                vertex_conflict_steps=vertex, edge_conflict_steps=edge))
    relevant = sum(p["relevant"] for p in pairs)
    errors = sum(p["fov_error"] for p in pairs)
    return dict(pairs=pairs, directed_pairs=len(pairs),
        ordering_errors=sum(p["ordering_error"] for p in pairs),
        relevant_pairs=relevant, fov_errors=errors,
        xi_fov=errors/max(1, relevant),
        xi_aoi=sum(p["aoi_weighted_error"] for p in pairs)/max(1, relevant))
