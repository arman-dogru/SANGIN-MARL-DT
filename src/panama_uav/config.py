from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
import math

COMMUNICATION_SCHEMA = 4
FIXED_MAP = "maps/random-32-32-10.map"


class ExperimentMode(str, Enum):
    UAV = "uav_panama"

    @property
    def spatial_channels(self):
        return 3

    @property
    def policy_mode(self):
        return self


class LearningCondition(str, Enum):
    CLEAN = "clean"
    UPLINK = "uplink_delayed"
    DOWNLINK = "downlink_delayed"
    COMBINED = "combined"


@dataclass(frozen=True)
class MapConfig:
    fov_radius: int = 7
    astar_waypoints: int = 5
    communicated_path_steps: int = 5
    communication_range_factor: float = 2.0
    meters_per_cell: float = 1.0


@dataclass(frozen=True)
class RewardConfig:
    goal: float = 2.0
    time_step: float = -0.01
    collision_agent: float = -1.0
    collision_obstacle: float = -1.0
    distance_improvement: float = 0.23
    path_conflict: float = -1.0
    hold: float = 0.05
    active_yield: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    target_environment_steps: int = 80_000_000
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 2_000_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 5_000_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 60_000_000
    per_epsilon: float = 1e-5
    model_broadcast_steps: int = 16
    checkpoint_interval_steps: int = 10_000
    validation_interval_episodes: int = 250
    validation_episodes: int = 200
    budget_seconds: float = 10_800.0
    shutdown_reserve_seconds: float = 60.0


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    map_path: str = FIXED_MAP
    num_agents: int = 2
    max_episode_steps: int = 250
    min_episodes: int = 500
    max_episodes: int = 200_000
    graduation_success_rate: float = 0.95


@dataclass(frozen=True)
class UAVConfig:
    environment: str = "urban"
    carrier_frequency_hz: float = 2.5e9
    max_path_loss_db: float = 85.0
    speed_m_per_step: float = 1.0
    minimum_altitude_m: float = 20.0
    blackout_threshold_db: float = -8.47


@dataclass(frozen=True)
class TransportConfig:
    profile: str = "controlled"
    mission_route: str = "uav"
    stale_information: tuple[str, ...] = ()
    seconds_per_step: float = 1.0
    delay_steps: float = 5.0
    windows_min: int = 1
    windows_max: int = 2
    reservation_steps: int = 2
    access_uplink_mbps: float = 50.0
    access_downlink_mbps: float = 50.0
    satellite_uplink_mbps: float = 10.0
    satellite_downlink_mbps: float = 50.0
    one_way_propagation_seconds: float = 0.025
    handover_interval_seconds: float = 15.0
    handover_interruption_seconds: float = 0.05
    handover_enabled: bool = True
    contention_enabled: bool = True
    edge_processing_seconds: float = 0.0
    cloud_processing_seconds: float = 0.0
    uplink_trace: str | None = None
    downlink_trace: str | None = None
    leo_timeline: str | None = None
    backhaul_time_offset_seconds: float = 0.0
    orbital_replay: bool = False
    packet_payload_bytes: int = 0
    packet_overhead_bytes: int = 48
    scheduler_frame_seconds: float = 0.02
    scheduler_guard_fraction: float = 0.05
    log_service_chunks: bool = False
    model_distribution: str = "per_agent"

    def __post_init__(self):
        if self.profile not in ("perfect", "controlled", "physical"):
            raise ValueError("Unknown communication profile")
        if self.mission_route not in ("uav", "cloud"):
            raise ValueError("Unknown mission route")
        if set(self.stale_information) - {"task", "intentions", "priorities"}:
            raise ValueError("Unknown stale information")
        if self.seconds_per_step <= 0 or self.delay_steps < 0:
            raise ValueError("Invalid simulated time")
        if self.backhaul_time_offset_seconds < 0:
            raise ValueError("Backhaul timeline offset must be nonnegative")
        if self.packet_payload_bytes < 0 or self.packet_overhead_bytes < 0:
            raise ValueError("Invalid packet sizes")
        if self.scheduler_frame_seconds <= 0 or not 0 <= self.scheduler_guard_fraction < 1:
            raise ValueError("Invalid scheduling frame")
        if self.orbital_replay and not self.leo_timeline:
            raise ValueError("Orbital replay requires an archived timeline")
        if self.model_distribution not in ("per_agent", "uav_fanout"):
            raise ValueError("Unknown model distribution route")
        if self.leo_timeline and (self.uplink_trace or self.downlink_trace):
            raise ValueError(
                "Use orbital modeling or empirical delay traces separately; do not double-count"
            )
        if not 1 <= self.windows_min <= self.windows_max <= 2:
            raise ValueError("Use one or two reservation windows")
        if self.reservation_steps < 1:
            raise ValueError("Invalid reservation duration")
        if (
            min(
                self.access_uplink_mbps,
                self.access_downlink_mbps,
                self.satellite_uplink_mbps,
                self.satellite_downlink_mbps,
            )
            <= 0
        ):
            raise ValueError("Capacity must be positive")
        if (
            self.one_way_propagation_seconds < 0
            or self.handover_interval_seconds <= 0
            or self.handover_interruption_seconds < 0
        ):
            raise ValueError("Invalid physical delays")


@dataclass(frozen=True)
class EvaluationConfig:
    agent_counts: tuple[int, ...] = (2, 4, 6, 8)
    cases_per_count: int = 50
    delay_steps: tuple[int, ...] = (1, 2, 5, 10, 20)
    bootstrap_samples: int = 2000


@dataclass(frozen=True)
class ResourceAllocationConfig:
    enabled: bool = False
    task_input_bits: int = 100_000
    cycles_per_bit: float = 1000.0
    drone_cycles_per_second: float = 1e9
    cloud_cycles_per_second: float = 1e10
    robot_power_w: float = 0.2
    drone_power_w: float = 2.0
    weight_min: float = 1.0
    weight_max: float = 4.0
    policy: str = "priority"
    energy_coefficient: float = 1.0
    latency_coefficient: float = 1.0
    access_bandwidth_hz: float = 20e6
    noise_psd_w_hz: float = 3.9810717055e-21  # -174 dBm/Hz, assumed thermal floor
    access_noise_figure_db: float = 7.0
    terminal_eirp_dbw: float = 35.0
    a2s_frequency_hz: float = 14.25e9
    a2s_bandwidth_hz: float = 20e6
    assumed_satellite_distance_km: float = 550.0
    assumed_gateway_distance_km: float = 550.0
    assumed_atmosphere_db: float = 3.0
    satellite_receive_gt_db_k: float = -5.0
    implementation_loss_db: float = 3.0
    shannon_efficiency: float = 0.5
    decision_budget_seconds: float = 0.1
    age_reference_seconds: float = 1.0
    failure_directory: str = "output/allocation_failures"

    def __post_init__(self):
        positive = (self.task_input_bits, self.cycles_per_bit, self.drone_cycles_per_second,
                    self.cloud_cycles_per_second, self.robot_power_w, self.drone_power_w,
                    self.weight_min, self.weight_max, self.access_bandwidth_hz,
                    self.noise_psd_w_hz, self.age_reference_seconds, self.latency_coefficient)
        positive += (self.a2s_frequency_hz, self.a2s_bandwidth_hz, self.assumed_satellite_distance_km,
                     self.assumed_gateway_distance_km, self.shannon_efficiency)
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Resource parameters must be finite and positive")
        if self.weight_max < self.weight_min or self.policy not in ("equal", "static", "aoi", "priority"):
            raise ValueError("Invalid resource weights or policy")
        for v in (self.energy_coefficient, self.decision_budget_seconds, self.access_noise_figure_db,
                  self.assumed_atmosphere_db, self.implementation_loss_db):
            if not math.isfinite(v) or v < 0:
                raise ValueError("Invalid resource coefficient or control budget")
        if not all(math.isfinite(v) for v in (self.terminal_eirp_dbw, self.satellite_receive_gt_db_k)):
            raise ValueError("EIRP and receiver G/T must be finite")
        if self.shannon_efficiency > 1:
            raise ValueError("Shannon efficiency must not exceed one")


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    map: MapConfig = field(default_factory=MapConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: tuple[CurriculumStage, ...] = field(
        default_factory=lambda: tuple(
            CurriculumStage(f"agents-{n}", num_agents=n) for n in (2, 4, 6, 8)
        )
    )
    uav: UAVConfig = field(default_factory=UAVConfig)
    communication: TransportConfig = field(default_factory=TransportConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    resource_allocation: ResourceAllocationConfig = field(default_factory=ResourceAllocationConfig)

    def resolve(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.project_root / path

    @property
    def vector_length(self):
        return 2 + 2 * self.map.astar_waypoints

    def profile(self, name):
        if name == "full":
            return self
        if name != "smoke":
            raise ValueError(name)
        return replace(
            self,
            training=replace(
                self.training,
                target_environment_steps=256,
                batch_size=8,
                replay_capacity=512,
                checkpoint_interval_steps=32,
                budget_seconds=120,
                shutdown_reserve_seconds=5,
            ),
        )

    def to_dict(self):
        result = asdict(self)
        result["project_root"] = str(self.project_root)
        return result

    @classmethod
    def from_dict(cls, values):
        """Restore a manifest, including optional resource settings in new runs."""
        communication = dict(values["communication"])
        communication["stale_information"] = tuple(communication["stale_information"])
        evaluation = dict(values["evaluation"])
        for name in ("agent_counts", "delay_steps"):
            evaluation[name] = tuple(evaluation[name])
        return cls(Path(values["project_root"]), MapConfig(**values["map"]),
            RewardConfig(**values["reward"]), TrainingConfig(**values["training"]),
            tuple(CurriculumStage(**stage) for stage in values["curriculum"]),
            UAVConfig(**values["uav"]), TransportConfig(**communication), EvaluationConfig(**evaluation),
            ResourceAllocationConfig(**values.get("resource_allocation", {})))


def load_config(path="configs/base.toml"):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    allowed = {
        "map",
        "reward",
        "training",
        "curriculum",
        "uav",
        "communication",
        "evaluation",
        "resource_allocation",
    }
    if set(raw) - allowed:
        raise ValueError(
            "Legacy or unknown configuration sections; use a new communication-study config"
        )
    communication = dict(raw.get("communication", {}))
    if "stale_information" in communication:
        communication["stale_information"] = tuple(communication["stale_information"])
    evaluation = dict(raw.get("evaluation", {}))
    for key in ("agent_counts", "delay_steps"):
        if key in evaluation:
            evaluation[key] = tuple(evaluation[key])
    stages = tuple(
        CurriculumStage(**s) for s in raw.get("curriculum", {}).get("stages", [])
    )
    config = ProjectConfig(
        path.parent.parent,
        MapConfig(**raw.get("map", {})),
        RewardConfig(**raw.get("reward", {})),
        TrainingConfig(**raw.get("training", {})),
        stages or ProjectConfig(path.parent.parent).curriculum,
        UAVConfig(**raw.get("uav", {})),
        TransportConfig(**communication),
        EvaluationConfig(**evaluation),
        ResourceAllocationConfig(**raw.get("resource_allocation", {})),
    )
    if config.resource_allocation.enabled:
        selected = set(config.communication.stale_information)
        if ("intentions" in selected) != ("priorities" in selected):
            raise ValueError("Paper mode requires intentions and priorities to be delayed together")
        if config.communication.uplink_trace or config.communication.downlink_trace:
            raise ValueError("Paper energy/service accounting requires modeled rates, not delay traces")
    if tuple(s.num_agents for s in config.curriculum) != (2, 4, 6, 8) or any(
        s.map_path != FIXED_MAP
        or s.max_episode_steps != 250
        or s.min_episodes < 500
        or s.graduation_success_rate < 0.95
        for s in config.curriculum
    ):
        raise ValueError(
            "Study requires fixed map, phases 2/4/6/8, 250 steps, minimum 500 episodes and 95% qualification"
        )
    if (
        config.training.validation_episodes != 200
        or config.training.validation_interval_episodes != 250
        or config.training.model_broadcast_steps != 16
    ):
        raise ValueError(
            "Study requires 200 validation episodes every 250 episodes, snapshots every 16 steps"
        )
    if (
        config.reward != RewardConfig()
        or replace(config.map, meters_per_cell=1.0) != MapConfig()
    ):
        raise ValueError("This study retains 0.1 task rewards and observations")
    if not 0 < config.training.budget_seconds <= 86400:
        raise ValueError(
            "A condition allocation must be positive and at most 24 hours"
        )
    return config
