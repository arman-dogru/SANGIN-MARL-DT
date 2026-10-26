"""Physical conflict detection and local post-completion control.

The resolver receives delivered DT priorities explicitly. It never queries a
physical agent's priority. The local controller sees only its square FoV and
network-delivered intentions.
"""
from collections import defaultdict

DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}


def conflicts(before, targets):
    events = []
    cells = defaultdict(list)
    for robot, target in targets.items():
        cells[tuple(target)].append(robot)
    for cell, robots in cells.items():
        if len(robots) > 1:
            events.append(dict(kind="vertex", robots=sorted(robots), cell=list(cell)))
    for first in sorted(targets):
        for second in sorted(targets):
            if first < second and targets[first] == before[second] and targets[second] == before[first]:
                events.append(dict(kind="edge_swap", robots=[first, second],
                    edge=[list(before[first]), list(before[second])]))
    return events


def resolve_delivered(before, intended, priorities):
    """Optional post-conflict controller; stationary occupants always retain cells."""
    final = dict(intended)
    while True:
        events = conflicts(before, final)
        if not events:
            return final
        # Each pass stops at least one moving robot, so this also handles cascades.
        for event in events:
            robots = event["robots"]
            if event["kind"] == "edge_swap":
                stop = robots
            else:
                stayers = [r for r in robots if final[r] == before[r]]
                winner = min(stayers or robots, key=lambda r: (priorities[r], r))
                stop = [r for r in robots if r != winner]
            for robot in stop:
                final[robot] = before[robot]


def completed_action(report, cached, own_priority):
    """Yield from delivered active intentions, then return using local free cells."""
    radius = len(report.obstacles)//2
    row, col = report.position
    threatened = set()
    sources = {report.robot_id: report.sensed_at}
    for peer, cache in cached.items():
        state = cache.value
        if peer == report.robot_id:
            continue
        sources.update({int(k): min(sources.get(int(k), float(t)), float(t))
                        for k, t in cache.source_timestamps.items()})
        if not state["completed"] and state["priority"] < own_priority:
            threatened.update(tuple(p) for p in state["path"][1:]
                if max(abs(p[0]-row), abs(p[1]-col)) <= radius)
    choices = []
    for action, (dr, dc) in DELTAS.items():
        if abs(dr) > radius or abs(dc) > radius:
            continue
        point = (row+dr, col+dc)
        if report.obstacles[radius+dr][radius+dc] or report.occupancy[radius+dr][radius+dc]:
            continue
        if point in threatened:
            continue
        distance = abs(point[0]-report.goal[0])+abs(point[1]-report.goal[1])
        # Prefer remaining still on ties to avoid post-goal wandering.
        choices.append((distance, action != 4, action))
    action = min(choices)[-1] if choices else 4
    return action, sources, dict(threatened_cells=[list(p) for p in sorted(threatened)],
        local_position=list(report.position), returning=report.position != tuple(report.goal))


def cached_path_action(report, cached, horizon=5):
    """Follow a delivered own-path successor, yielding to delivered peer priorities.

    Local sensing supplies pose, obstacles and occupancy only. No pathfinding,
    DQN inference, DT refresh or peer-truth priority query occurs here. A stale
    path that no longer contains the robot cannot authorize a movement.
    """
    own = cached[report.robot_id]
    state = own.value
    point = tuple(report.position)
    path = [tuple(cell) for cell in state["path"]]
    sources = {}
    for cache in cached.values():
        for robot, timestamp in cache.source_timestamps.items():
            robot, timestamp = int(robot), float(timestamp)
            sources[robot] = min(sources.get(robot, timestamp), timestamp)
    details = dict(local_position=list(point), local_sensed_at=report.sensed_at,
        self_cache_version=own.version, self_generation_time=own.timestamp,
        cached_priority=state["priority"], path_index=None, candidate_action=4,
        target=None, blocking_peers=[], reason="cached_path_unavailable")
    if not path or point not in path or state["completed"]:
        return 4, sources, details
    index = path.index(point)
    details["path_index"] = index
    if index+1 >= len(path):
        details["reason"] = "cached_path_exhausted"
        return 4, sources, details
    target = path[index+1]
    delta = target[0]-point[0], target[1]-point[1]
    action = next((a for a, step in DELTAS.items() if step == delta and a != 4), None)
    details["target"] = list(target)
    if action is None:
        details["reason"] = "cached_step_not_adjacent"
        return 4, sources, details
    details["candidate_action"] = action
    radius = len(report.obstacles)//2
    dr, dc = delta
    if max(abs(dr), abs(dc)) > radius or report.obstacles[radius+dr][radius+dc]:
        details["reason"] = "local_obstacle"
        return 4, sources, details
    if report.occupancy[radius+dr][radius+dc]:
        details["reason"] = "local_occupancy"
        return 4, sources, details
    for peer, cache in sorted(cached.items()):
        other = cache.value
        if peer == report.robot_id or other["completed"] or other["priority"] >= state["priority"]:
            continue
        intentions = [tuple(cell) for cell in other["path"][1:horizon+1]
                      if max(abs(cell[0]-point[0]), abs(cell[1]-point[1])) <= radius]
        if target in intentions:
            details["blocking_peers"].append(peer)
    if details["blocking_peers"]:
        details["reason"] = "higher_priority_cached_intention"
        return 4, sources, details
    details["reason"] = "cached_successor"
    return action, sources, details
