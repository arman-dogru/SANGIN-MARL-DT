from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import UAVConfig
from .radio import RadioSnapshot, normalize_quality

ENVIRONMENT_PARAMETERS: dict[str, tuple[float, float, float, float]] = {
    "suburban": (4.88, 0.43, 0.1, 21.0),
    "urban": (9.61, 0.16, 1.0, 20.0),
    "dense_urban": (12.08, 0.11, 1.6, 23.0),
    "highrise": (27.23, 0.08, 2.3, 34.0),
}


class AirToGroundModel:
    def __init__(self, config: UAVConfig) -> None:
        if config.environment not in ENVIRONMENT_PARAMETERS:
            raise ValueError(f"Unknown UAV environment: {config.environment}")
        self.config = config
        self.a, self.b, self.eta_los, self.eta_nlos = ENVIRONMENT_PARAMETERS[
            config.environment
        ]
        self.alpha_opt = self._find_optimal_alpha()
        self.maximum_radius_m = self.radius_for_alpha(self.alpha_opt)
        self.optimal_height_m = self.alpha_opt * self.maximum_radius_m

    def los_probability(self, alpha: np.ndarray | float) -> np.ndarray | float:
        angle_deg = np.degrees(np.arctan(alpha))
        return 1.0 / (1.0 + self.a * np.exp(-self.b * (angle_deg - self.a)))

    def radius_for_alpha(self, alpha: float) -> float:
        c = 3e8
        constant = (
            20.0 * math.log10(4.0 * math.pi * self.config.carrier_frequency_hz / c)
            + self.eta_nlos
        )
        excess = (self.eta_los - self.eta_nlos) * float(self.los_probability(alpha))
        numerator = 10.0 ** ((self.config.max_path_loss_db - constant - excess) / 10.0)
        return math.sqrt(numerator / (1.0 + alpha**2))

    def _log_radius_derivative(self, alpha: float) -> float:
        probability = float(self.los_probability(alpha))
        d_probability = (
            probability
            * (1.0 - probability)
            * self.b
            * (180.0 / math.pi)
            / (1.0 + alpha**2)
        )
        a_loss = self.eta_los - self.eta_nlos
        return -(math.log(10.0) / 10.0) * a_loss * d_probability - (
            2.0 * alpha / (1.0 + alpha**2)
        )

    def _find_optimal_alpha(self) -> float:
        low, high = 1e-6, math.tan(math.radians(89.9))
        low_value, high_value = (
            self._log_radius_derivative(low),
            self._log_radius_derivative(high),
        )
        if low_value <= 0 or high_value >= 0:
            raise RuntimeError("Could not bracket the air-to-ground optimum")
        for _ in range(100):
            mid = (low + high) / 2.0
            value = self._log_radius_derivative(mid)
            if abs(value) < 1e-12:
                return mid
            if value > 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    def path_loss_db(
        self, height_m: float, horizontal_distance_m: np.ndarray | float
    ) -> np.ndarray:
        horizontal = np.asarray(horizontal_distance_m, dtype=np.float64)
        height = max(float(height_m), 1e-6)
        alpha = height / np.maximum(horizontal, 1e-6)
        probability = np.asarray(self.los_probability(alpha))
        distance_3d = np.sqrt(horizontal**2 + height**2)
        free_space = 20.0 * np.log10(
            4.0 * math.pi * self.config.carrier_frequency_hz / 3e8
        ) + 20.0 * np.log10(np.maximum(distance_3d, 1e-6))
        excess = probability * self.eta_los + (1.0 - probability) * self.eta_nlos
        return free_space + excess

    def effective_snr_db(
        self, height_m: float, horizontal_distance_m: np.ndarray | float
    ) -> np.ndarray:
        path_loss = self.path_loss_db(height_m, horizontal_distance_m)
        return self.config.blackout_threshold_db + (
            self.config.max_path_loss_db - path_loss
        )


@dataclass(frozen=True)
class UAVStepDiagnostics:
    target_position: np.ndarray
    distance_moved: float
    oracle_served: int
    actual_served: int
    tracking_error_m: float


class UAVController:
    def __init__(
        self, map_shape: tuple[int, int], meters_per_cell: float, config: UAVConfig
    ) -> None:
        self.map_shape = map_shape
        self.meters_per_cell = meters_per_cell
        self.config = config
        self.model = AirToGroundModel(config)
        height, width = map_shape
        self.map_width_m = width * meters_per_cell
        self.map_height_m = height * meters_per_cell
        self.position = np.array(
            [
                self.map_width_m / 2.0,
                self.map_height_m / 2.0,
                self.model.optimal_height_m,
            ],
            dtype=np.float64,
        )
        self.target_position = self.position.copy()
        self.total_distance_m = 0.0

    def reset(self) -> None:
        self.position[:] = (
            self.map_width_m / 2.0,
            self.map_height_m / 2.0,
            self.model.optimal_height_m,
        )
        self.target_position = self.position.copy()
        self.total_distance_m = 0.0

    def optimal_target(self, points_xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_xy_m, dtype=np.float64)
        if points.size == 0:
            return self.position.copy(), np.zeros(0, dtype=bool)
        radius = self.model.maximum_radius_m
        candidates: list[np.ndarray] = [point.copy() for point in points]
        for first in range(len(points)):
            for second in range(first + 1, len(points)):
                p1, p2 = points[first], points[second]
                delta = p2 - p1
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-12 or distance > 2.0 * radius:
                    continue
                midpoint = (p1 + p2) / 2.0
                offset_length = math.sqrt(max(radius**2 - (distance / 2.0) ** 2, 0.0))
                perpendicular = np.array([-delta[1], delta[0]]) / distance
                candidates.extend(
                    (
                        midpoint + offset_length * perpendicular,
                        midpoint - offset_length * perpendicular,
                    )
                )

        best_key: tuple[float, float, float, float, float] | None = None
        best_center = points[0]
        best_mask = np.zeros(len(points), dtype=bool)
        for center in candidates:
            distances = np.linalg.norm(points - center, axis=1)
            covered = distances <= radius + 1e-8
            count = int(covered.sum())
            required_radius = float(distances[covered].max()) if count else math.inf
            movement = float(np.linalg.norm(center - self.position[:2]))
            key = (
                -count,
                required_radius,
                movement,
                float(center[0]),
                float(center[1]),
            )
            if best_key is None or key < best_key:
                best_key, best_center, best_mask = key, center, covered

        required_radius = (
            float(np.linalg.norm(points[best_mask] - best_center, axis=1).max())
            if best_mask.any()
            else 0.0
        )
        target_height = max(
            self.config.minimum_altitude_m, self.model.alpha_opt * required_radius
        )
        target_height = min(target_height, self.model.optimal_height_m)
        return np.array(
            [best_center[0], best_center[1], target_height], dtype=np.float64
        ), best_mask

    def update(self, points_xy_m: np.ndarray) -> UAVStepDiagnostics:
        target, oracle_mask = self.optimal_target(points_xy_m)
        self.target_position = target
        direction = target - self.position
        distance = float(np.linalg.norm(direction))
        moved = min(self.config.speed_m_per_step, distance)
        if distance > 1e-12:
            self.position += direction * (moved / distance)
        self.total_distance_m += moved
        points = np.asarray(points_xy_m, dtype=np.float64)
        if points.size:
            horizontal = np.linalg.norm(points - self.position[:2], axis=1)
            actual = (
                self.model.effective_snr_db(self.position[2], horizontal)
                >= self.config.blackout_threshold_db
            )
        else:
            actual = np.zeros(0, dtype=bool)
        return UAVStepDiagnostics(
            target_position=target.copy(),
            distance_moved=moved,
            oracle_served=int(oracle_mask.sum()),
            actual_served=int(actual.sum()),
            tracking_error_m=float(np.linalg.norm(target - self.position)),
        )

    def snapshot(self) -> RadioSnapshot:
        height, width = self.map_shape
        ys = (np.arange(height, dtype=np.float64) + 0.5) * self.meters_per_cell
        xs = (np.arange(width, dtype=np.float64) + 0.5) * self.meters_per_cell
        x_grid, y_grid = np.meshgrid(xs, ys)
        horizontal = np.sqrt(
            (x_grid - self.position[0]) ** 2 + (y_grid - self.position[1]) ** 2
        )
        snr = self.model.effective_snr_db(self.position[2], horizontal)
        normalized = normalize_quality(snr, self.config.blackout_threshold_db, 30.0)
        return RadioSnapshot(snr.astype(np.float32), normalized)

    def positions_to_xy_m(
        self, positions: list[tuple[int, int]] | tuple[tuple[int, int], ...]
    ) -> np.ndarray:
        return np.asarray(
            [
                [(col + 0.5) * self.meters_per_cell, (row + 0.5) * self.meters_per_cell]
                for row, col in positions
            ],
            dtype=np.float64,
        )
