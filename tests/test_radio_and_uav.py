from pathlib import Path

import numpy as np

from panama_uav.config import load_config
from panama_uav.uav import AirToGroundModel, UAVController

ROOT = Path(__file__).resolve().parents[1]


def test_retained_uav_geometry():
    c = load_config(ROOT / "configs/base.toml")
    model = AirToGroundModel(c.uav)
    assert abs(model.alpha_opt - 0.9143603) < 1e-6
    assert abs(model.maximum_radius_m - 100.5848753) < 1e-5
    controller = UAVController((300, 300), 1.0, c.uav)
    before = controller.position.copy()
    diagnostic = controller.update(np.array([[0.5, 0.5], [299.5, 299.5]]))
    assert np.linalg.norm(controller.position - before) <= 1 + 1e-9
    assert diagnostic.oracle_served >= diagnostic.actual_served
