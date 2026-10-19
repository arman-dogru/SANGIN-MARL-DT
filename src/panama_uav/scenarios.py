from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Scenario:
    path: Path
    map_path: Path
    starts: tuple[tuple[int, int], ...]
    goals: tuple[tuple[int, int], ...]

    @property
    def name(self) -> str:
        return self.path.stem

    def first(self, num_agents: int) -> Scenario:
        if num_agents > len(self.starts):
            raise ValueError(
                f"{self.path} has {len(self.starts)} agents, requested {num_agents}"
            )
        return Scenario(
            self.path, self.map_path, self.starts[:num_agents], self.goals[:num_agents]
        )


def load_map(path: str | Path) -> np.ndarray:
    path = Path(path)
    lines = [line.rstrip("\n") for line in path.read_text().splitlines()]
    try:
        start = lines.index("map") + 1
    except ValueError as exc:
        raise ValueError(f"Invalid MovingAI map without 'map' marker: {path}") from exc
    rows = lines[start:]
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError(f"Invalid or ragged map: {path}")
    grid = np.array(
        [[1 if char in "@TO#" else 0 for char in row] for row in rows], dtype=np.uint8
    )
    return grid


def load_scenario(
    path: str | Path, maps_dir: str | Path, num_agents: int | None = None
) -> Scenario:
    path = Path(path)
    rows: list[tuple[tuple[int, int], tuple[int, int]]] = []
    map_name: str | None = None
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 9 or not parts[0].isdigit():
            continue
        map_name = map_name or parts[1]
        rows.append(((int(parts[5]), int(parts[4])), (int(parts[7]), int(parts[6]))))
        if num_agents is not None and len(rows) >= num_agents:
            break
    if not rows or map_name is None:
        raise ValueError(f"No agent rows found in {path}")
    map_filename = map_name if map_name.endswith(".map") else f"{map_name}.map"
    map_path = Path(maps_dir) / Path(map_filename).name
    if not map_path.exists():
        raise FileNotFoundError(f"Scenario map not found: {map_path}")
    starts, goals = zip(*rows)
    return Scenario(path.resolve(), map_path.resolve(), tuple(starts), tuple(goals))


def discover_scenarios(
    directories: Iterable[str | Path], maps_dir: str | Path
) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.scen")):
            scenarios.append(load_scenario(path, maps_dir))
    return scenarios
