"""Independent bounded conflict-based MAPF search for diagnostic feasibility.

This planner is deliberately independent of PANAMA priorities and its collision
resolver. It uses synchronous cardinal moves/WAIT, forbids vertex conflicts and
opposite edge swaps, and holds each robot at its final goal. A valid solution is
a constructive feasibility witness; a timeout is never an infeasibility claim.
"""
from collections import deque
import heapq
import itertools
import time


class _Limit(Exception):
    pass


def validate_paths(grid, starts, goals, paths, horizon=250):
    if len(paths) != len(starts) or len(goals) != len(starts):
        return False
    for start, goal, path in zip(starts, goals, paths):
        if not path or tuple(path[0]) != tuple(start) or tuple(path[-1]) != tuple(goal):
            return False
        if len(path)-1 > horizon:
            return False
        for point in path:
            r, c = point
            if not (0 <= r < len(grid) and 0 <= c < len(grid[0])) or grid[r][c] != 0:
                return False
        if any(abs(a[0]-b[0])+abs(a[1]-b[1]) > 1 for a, b in zip(path, path[1:])):
            return False
    return first_conflict(paths) is None


def first_conflict(paths):
    """Earliest vertex/edge conflict, padding completed paths by holding goals."""
    at = lambda i, t: tuple(paths[i][min(t, len(paths[i])-1)])
    for tick in range(max(map(len, paths))):
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                if at(i, tick) == at(j, tick):
                    return ('vertex', i, j, tick, at(i, tick), at(i, tick))
                if tick and at(i, tick-1) == at(j, tick) and at(j, tick-1) == at(i, tick):
                    return ('edge', i, j, tick, at(i, tick-1), at(i, tick))
    return None


def solve_mapf(grid, starts, goals, *, horizon=250, timeout_seconds=2., node_limit=10000):
    """CBS with exact space-time A* low-level searches within a finite horizon.

    ``infeasible_within_horizon`` is a finite-horizon proof only. ``timeout`` and
    ``node_limit`` leave feasibility unresolved. Search is cost ordered, but this
    interface makes no research claim about reference optimality or efficiency.
    """
    begin = time.monotonic(); deadline = begin+timeout_seconds
    starts = [tuple(p) for p in starts]; goals = [tuple(p) for p in goals]
    expanded = 0; low_level_expanded = 0
    rows, cols = len(grid), len(grid[0])
    def valid(p):
        return 0 <= p[0] < rows and 0 <= p[1] < cols and grid[p[0]][p[1]] == 0
    def finish(status, paths=None):
        return dict(status=status, wall_seconds=time.monotonic()-begin,
            high_level_expanded=expanded, low_level_expanded=low_level_expanded,
            horizon=horizon, timeout_seconds=timeout_seconds, node_limit=node_limit,
            goal_convention='stay at final goal; vertex and opposite-edge conflicts forbidden',
            paths=paths, makespan=max(map(len, paths))-1 if paths else None,
            sum_of_costs=sum(len(p)-1 for p in paths) if paths else None)
    if not starts or len(starts) != len(goals) or len(set(starts)) != len(starts):
        raise ValueError('Distinct starts and equal nonempty start/goal lists required')
    if horizon < 0 or timeout_seconds < 0 or node_limit < 0:
        raise ValueError('Nonnegative search bounds required')
    if len(set(goals)) != len(goals) or any(not valid(p) for p in starts+goals):
        return finish('infeasible_static')
    neighbors = {}
    for r in range(rows):
        for c in range(cols):
            if valid((r, c)):
                neighbors[r, c] = [p for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1),(r,c)) if valid(p)]
    distances = []
    for goal in goals:
        distances.append({goal:0}); queue = deque([goal])
        while queue:
            p = queue.popleft()
            for q in neighbors[p]:
                if q not in distances[-1]:
                    distances[-1][q] = distances[-1][p]+1; queue.append(q)
    if any(p not in distances[i] for i, p in enumerate(starts)):
        return finish('infeasible_static')
    def low_level(robot, constraints):
        nonlocal low_level_expanded
        vertices = {(t,p) for who,kind,t,p,q in constraints if who == robot and kind == 'vertex'}
        edges = {(t,p,q) for who,kind,t,p,q in constraints if who == robot and kind == 'edge'}
        if (0, starts[robot]) in vertices:
            return None
        last_goal_constraint = max((t for t,p in vertices if p == goals[robot]), default=-1)
        serial = itertools.count(); start = (starts[robot],0)
        frontier = [(distances[robot][starts[robot]],next(serial),start)]
        parent = {start:None}
        while frontier:
            if time.monotonic() >= deadline:
                raise _Limit
            _, _, (point,tick) = heapq.heappop(frontier); low_level_expanded += 1
            if point == goals[robot] and tick > last_goal_constraint:
                path = []; state = (point,tick)
                while state is not None:
                    path.append(state[0]); state = parent[state]
                return path[::-1]
            if tick == horizon:
                continue
            for target in neighbors[point]:
                state = (target,tick+1)
                if (tick+1,target) in vertices or (tick+1,point,target) in edges or state in parent:
                    continue
                distance = distances[robot].get(target, horizon+1)
                if tick+1+distance > horizon:
                    continue
                parent[state] = (point,tick)
                heapq.heappush(frontier,(tick+1+distance,next(serial),state))
        return None
    try:
        if time.monotonic() >= deadline:
            return finish('timeout')
        root = [low_level(i, ()) for i in range(len(starts))]
        if any(p is None for p in root):
            return finish('infeasible_within_horizon')
        serial = itertools.count(); frontier = [(sum(map(len,root)),next(serial),(),root)]
        seen = {()}
        while frontier:
            if time.monotonic() >= deadline:
                return finish('timeout')
            if expanded >= node_limit:
                return finish('node_limit')
            _, _, constraints, paths = heapq.heappop(frontier); expanded += 1
            conflict = first_conflict(paths)
            if conflict is None:
                assert validate_paths(grid,starts,goals,paths,horizon)
                return finish('feasible',paths)
            kind,i,j,tick,p,q = conflict
            for robot in (i,j):
                edge = (p,q) if robot == i else (q,p)
                addition = (robot,kind,tick,edge[0],edge[1])
                branch = tuple(sorted((*constraints,addition)))
                if branch in seen:
                    continue
                seen.add(branch); path = low_level(robot,branch)
                if path is not None:
                    update = list(paths); update[robot] = path
                    heapq.heappush(frontier,(sum(map(len,update)),next(serial),branch,update))
        return finish('infeasible_within_horizon')
    except _Limit:
        return finish('timeout')
