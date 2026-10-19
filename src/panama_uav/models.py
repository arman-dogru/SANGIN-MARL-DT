from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn


class DQN(nn.Module):
    """Dual-stream network compatible with the legacy PANAMA checkpoints."""

    def __init__(
        self,
        spatial_shape: tuple[int, int, int],
        vector_length: int,
        num_actions: int = 5,
    ) -> None:
        super().__init__()
        self.cnn_base = nn.Sequential(
            nn.Conv2d(spatial_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_size = int(self.cnn_base(torch.zeros(1, *spatial_shape)).shape[1])
        self.vector_processor = nn.Sequential(
            nn.Linear(vector_length, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(cnn_size + 32, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        spatial = self.cnn_base(observations["spatial"])
        vector = self.vector_processor(observations["vector"])
        return self.head(torch.cat((spatial, vector), dim=1))


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def observation_tensors(
    observation: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "spatial": torch.from_numpy(
            np.asarray(observation["spatial"], dtype=np.float32)
        )
        .unsqueeze(0)
        .to(device),
        "vector": torch.from_numpy(np.asarray(observation["vector"], dtype=np.float32))
        .unsqueeze(0)
        .to(device),
    }


def select_action(
    model: DQN,
    observation: dict[str, np.ndarray],
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, 5))
    with torch.no_grad():
        return int(model(observation_tensors(observation, device)).argmax(dim=1).item())


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def select_actions(models, versions, observations, epsilon, rng, device):
    """Batch inference by deployed version, preserving per-agent exploration RNG order."""
    actions = {}
    groups = {}
    for agent in observations:
        if rng.random() < epsilon:
            actions[agent] = int(rng.integers(0, 5))
        else:
            groups.setdefault(versions[agent], []).append(agent)
    with torch.no_grad():
        for version, agents in groups.items():
            values = {
                key: torch.as_tensor(
                    np.stack([observations[a][key] for a in agents]), device=device
                )
                for key in ("spatial", "vector")
            }
            selected = models[version](values).argmax(1).cpu().tolist()
            actions.update(zip(agents, selected))
    return actions
