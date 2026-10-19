from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadioSnapshot:
    quality_db: np.ndarray
    normalized: np.ndarray

    def at_positions(
        self, positions: list[tuple[int, int]] | tuple[tuple[int, int], ...]
    ) -> np.ndarray:
        if not positions:
            return np.empty(0, dtype=np.float64)
        rows, cols = zip(*positions)
        return self.quality_db[np.asarray(rows), np.asarray(cols)]


def normalize_quality(
    values_db: np.ndarray, threshold_db: float, maximum_db: float
) -> np.ndarray:
    span = maximum_db - threshold_db
    if span <= 0:
        raise ValueError("normalization_max_db must exceed blackout_threshold_db")
    return np.clip((values_db - threshold_db) / span, 0.0, 1.0).astype(np.float32)
