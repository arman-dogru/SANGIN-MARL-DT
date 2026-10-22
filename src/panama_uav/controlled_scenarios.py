"""Versioned deterministic maps and interacting routes, never training inputs."""
from pathlib import Path
import numpy as np

from .environment import astar_path
from .records import sha256
from .scenarios import Scenario

FAMILIES = ('intersection', 'narrow_corridor', 'shared_bottleneck')


def geometry(family):
    grid = np.ones((32, 32), dtype=np.uint8)
    if family == 'intersection':
        grid[15:18, 3:29] = grid[3:29, 15:18] = 0
        starts = [(16,12),(12,16),(16,20),(20,16),(15,11),(11,15),(17,21),(21,17)]
        goals = [(16,20),(20,16),(16,12),(12,16),(15,21),(21,15),(17,11),(11,17)]
    elif family == 'narrow_corridor':
        grid[12:21, 4:13] = grid[12:21, 19:28] = 0
        grid[16, 4:28] = 0
        grid[15, 15] = grid[17, 17] = 0  # Passing bays; no physical rule changes.
        starts = [(16,11),(16,20),(15,11),(17,20),(17,11),(15,20),(14,11),(18,20)]
        goals = [(16,20),(16,11),(17,20),(15,11),(15,20),(17,11),(18,20),(14,11)]
    elif family == 'shared_bottleneck':
        grid[8:25, 6:15] = grid[8:25, 16:26] = 0
        grid[16,15] = 0  # One shared door between two open rooms.
        starts = [(15,13),(17,13),(15,18),(17,18),(14,12),(18,12),(14,19),(18,19)]
        goals = [(17,18),(15,18),(17,13),(15,13),(18,19),(14,19),(18,12),(14,12)]
    else:
        raise ValueError(f'Unknown controlled scenario: {family}')
    return grid, starts, goals


def make_cases(directory, cases=2, counts=(2,4), families=FAMILIES):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for family in families:
        grid, starts, goals = geometry(family)
        for index in range(cases):
            turns = index % 4
            transformed = np.rot90(grid, turns)
            def point(cell):
                r, c = cell
                for _ in range(turns):
                    r, c = 31-c, r
                return [r, c]
            map_path = directory / f'{family}-rotation-{turns}.map'
            text = 'type octile\nheight 32\nwidth 32\nmap\n' + '\n'.join(
                ''.join('@' if v else '.' for v in row) for row in transformed) + '\n'
            map_path.write_text(text, encoding='utf-8')
            for count in counts:
                own_starts = [point(p) for p in starts[:count]]
                own_goals = [point(p) for p in goals[:count]]
                if len(set(map(tuple,own_starts))) != count or len(set(map(tuple,own_goals))) != count:
                    raise ValueError('Controlled endpoints must be unique')
                if any(not astar_path(transformed, tuple(a), tuple(b)) for a,b in zip(own_starts, own_goals)):
                    raise ValueError('Unreachable controlled endpoint')
                records.append(dict(case_id=f'{family}-n{count}-case{index:02d}', scenario=family,
                    case=index, num_agents=count, density=f'{count}-robots', rotation=turns,
                    seed=71_000_000 + FAMILIES.index(family)*10000 + count*100 + index,
                    starts=own_starts, goals=own_goals, map_path=str(map_path.resolve()),
                    map_sha256=sha256(map_path), orbital_offset_seconds=float(index*100)))
    return records


def scenario_for(case):
    return Scenario(Path(case['case_id']), Path(case['map_path']),
        tuple(map(tuple,case['starts'])), tuple(map(tuple,case['goals'])))
