from __future__ import annotations

import torch
from torch.nn import functional

from .models import DQN
from .replay import PrioritizedReplayBuffer, ReplayBatch


def _learn(
    policy: DQN,
    target: DQN,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplayBuffer,
    batch_size: int,
    gamma: float,
    tau: float,
    device: torch.device,
) -> float | None:
    batch: ReplayBatch | None = replay.sample(batch_size)
    if batch is None:
        return None
    states = {
        name: torch.as_tensor(values, dtype=torch.float32, device=device)
        for name, values in batch.states.items()
    }
    next_states = {
        name: torch.as_tensor(values, dtype=torch.float32, device=device)
        for name, values in batch.next_states.items()
    }
    actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device).unsqueeze(
        1
    )
    rewards = torch.as_tensor(
        batch.rewards, dtype=torch.float32, device=device
    ).unsqueeze(1)
    terminals = torch.as_tensor(
        batch.terminals, dtype=torch.bool, device=device
    ).unsqueeze(1)
    weights = torch.as_tensor(
        batch.weights, dtype=torch.float32, device=device
    ).unsqueeze(1)
    current_q = policy(states).gather(1, actions)
    with torch.no_grad():
        next_actions = policy(next_states).argmax(dim=1, keepdim=True)
        next_q = target(next_states).gather(1, next_actions)
        expected = rewards + (~terminals) * gamma * next_q
    errors = (expected - current_q).detach().abs().cpu().numpy().reshape(-1)
    loss = (
        weights * functional.smooth_l1_loss(current_q, expected, reduction="none")
    ).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    with torch.no_grad():
        for target_parameter, policy_parameter in zip(
            target.parameters(), policy.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(policy_parameter, alpha=tau)
    replay.update_priorities(batch.tree_indices, errors)
    return float(loss.item())
