from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Transition = tuple[dict[str, np.ndarray], int, float, dict[str, np.ndarray], bool]


class SumTree:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.write_index = 0
        self.size = 0
        self.max_priority = 1.0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def update(self, tree_index: int, priority: float) -> None:
        delta = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += delta

    def add(self, priority: float, transition: Transition) -> None:
        tree_index = self.write_index + self.capacity - 1
        self.data[self.write_index] = transition
        self.update(tree_index, float(priority))
        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.max_priority = max(self.max_priority, float(priority))

    def retrieve(self, value: float) -> tuple[int, float, Transition]:
        index = 0
        while True:
            left = 2 * index + 1
            if left >= len(self.tree):
                break
            if value <= self.tree[left]:
                index = left
            else:
                value -= self.tree[left]
                index = left + 1
        data_index = index - self.capacity + 1
        return index, float(self.tree[index]), self.data[data_index]


@dataclass(frozen=True)
class ReplayBatch:
    states: dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    next_states: dict[str, np.ndarray]
    terminals: np.ndarray
    tree_indices: np.ndarray
    weights: np.ndarray


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float,
        beta_start: float,
        beta_steps: int,
        epsilon: float,
        seed: int,
    ) -> None:
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = max(beta_steps, 1)
        self.epsilon = epsilon
        self.samples_drawn = 0
        self.rng = np.random.default_rng(seed)
        self.generated_at = np.zeros(capacity, dtype=np.float64)
        self.last_sample_generated_at = np.empty(0)

    def __len__(self) -> int:
        return self.tree.size

    def add(self, transition: Transition, generated_at: float = 0.0) -> None:
        self.generated_at[self.tree.write_index] = generated_at
        self.tree.add(self.tree.max_priority, transition)

    def sample(self, batch_size: int) -> ReplayBatch | None:
        if self.tree.size < batch_size or self.tree.total <= 0:
            return None
        beta = min(
            1.0,
            self.beta_start
            + self.samples_drawn * (1.0 - self.beta_start) / self.beta_steps,
        )
        segment = self.tree.total / batch_size
        indices: list[int] = []
        priorities: list[float] = []
        transitions: list[Transition] = []
        for index in range(batch_size):
            value = self.rng.uniform(segment * index, segment * (index + 1))
            tree_index, priority, transition = self.tree.retrieve(float(value))
            indices.append(tree_index)
            priorities.append(priority)
            transitions.append(transition)
        self.last_sample_generated_at = self.generated_at[
            np.asarray(indices) - self.tree.capacity + 1
        ].copy()
        self.samples_drawn += batch_size
        probabilities = np.asarray(priorities, dtype=np.float64) / self.tree.total
        weights = (self.tree.size * np.maximum(probabilities, 1e-12)) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, terminals = zip(*transitions)
        return ReplayBatch(
            states={
                "spatial": np.asarray(
                    [state["spatial"] for state in states], dtype=np.float32
                ),
                "vector": np.asarray(
                    [state["vector"] for state in states], dtype=np.float32
                ),
            },
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states={
                "spatial": np.asarray(
                    [state["spatial"] for state in next_states], dtype=np.float32
                ),
                "vector": np.asarray(
                    [state["vector"] for state in next_states], dtype=np.float32
                ),
            },
            terminals=np.asarray(terminals, dtype=np.bool_),
            tree_indices=np.asarray(indices, dtype=np.int64),
            weights=weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        priorities = (
            np.abs(np.asarray(errors).reshape(-1)) + self.epsilon
        ) ** self.alpha
        for tree_index, priority in zip(indices, priorities):
            self.tree.update(int(tree_index), float(priority))
            self.tree.max_priority = max(self.tree.max_priority, float(priority))

    def state_dict(self):
        # Preserve every sample, exact tree, write index and RNG. No checkpoint truncation.
        import copy

        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        import copy

        self.__dict__.update(copy.deepcopy(state))
