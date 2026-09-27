"""Tabular off-policy Q-learning agent."""

from __future__ import annotations

import numpy as np

from .grid_environment import State


class QLearningAgent:
    """Classic tabular Q-learning agent using an epsilon-greedy policy."""

    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((5, 5, 4), dtype=float)

    @staticmethod
    def _index(state: State):
        row, col = state
        return row - 1, col - 1

    def select_action(self, state: State) -> int:
        """Choose an action index with epsilon-greedy exploration.

        The greedy branch uses NumPy argmax, mirroring the implementation
        approach described in the original coursework report.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 4))

        row, col = self._index(state)
        return int(np.argmax(self.q_table[row, col]))

    def update(
        self,
        state: State,
        action_index: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """Apply the standard Q-learning Temporal Difference update."""
        row, col = self._index(state)
        next_row, next_col = self._index(next_state)

        current = self.q_table[row, col, action_index]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_row, next_col])

        self.q_table[row, col, action_index] = current + self.alpha * (target - current)

    def state_values(self) -> np.ndarray:
        """Return V(s)=max_a Q(s,a) for every grid cell."""
        return np.max(self.q_table, axis=2)

    def greedy_policy(self) -> np.ndarray:
        """Return the greedy action index for each state."""
        return np.argmax(self.q_table, axis=2)
