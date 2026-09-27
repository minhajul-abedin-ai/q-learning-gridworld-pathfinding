"""Grid World environment used for the Q-learning project.

Coordinates are 1-indexed as (row, column), matching the assignment report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

State = Tuple[int, int]


@dataclass(frozen=True)
class Transition:
    next_state: State
    reward: float
    done: bool


class GridWorld:
    """A deterministic 5x5 Grid World with obstacles and a jump state."""

    ACTIONS = ("N", "S", "E", "W")
    ACTION_DELTAS: Dict[str, State] = {
        "N": (-1, 0),
        "S": (1, 0),
        "E": (0, 1),
        "W": (0, -1),
    }

    def __init__(self) -> None:
        self.rows = 5
        self.cols = 5
        self.start_state: State = (2, 1)
        self.goal_state: State = (5, 5)
        self.jump_in: State = (2, 4)
        self.jump_out: State = (4, 4)
        self.obstacles = {(3, 3), (3, 4), (3, 5), (4, 3)}

        self.step_penalty = -1.0
        self.jump_reward = 5.0
        self.goal_reward = 10.0
        self.state = self.start_state

    def reset(self) -> State:
        """Reset the environment to the starting state."""
        self.state = self.start_state
        return self.state

    def is_valid_state(self, state: State) -> bool:
        row, col = state
        inside = 1 <= row <= self.rows and 1 <= col <= self.cols
        return inside and state not in self.obstacles

    def step(self, action: str) -> Transition:
        """Apply one deterministic action and return the resulting transition."""
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action {action!r}. Expected one of {self.ACTIONS}.")

        row, col = self.state
        d_row, d_col = self.ACTION_DELTAS[action]
        candidate = (row + d_row, col + d_col)

        # Invalid moves leave the agent in the same state and still cost one point.
        if not self.is_valid_state(candidate):
            return Transition(self.state, self.step_penalty, False)

        # Entering the jump cell automatically teleports the agent.
        if candidate == self.jump_in:
            self.state = self.jump_out
            return Transition(self.state, self.jump_reward, False)

        # Reaching the goal ends the episode.
        if candidate == self.goal_state:
            self.state = self.goal_state
            return Transition(self.state, self.goal_reward, True)

        self.state = candidate
        return Transition(self.state, self.step_penalty, False)

    def layout_rows(self):
        """Return a row-wise representation suitable for CSV export."""
        rows = []
        for row in range(1, self.rows + 1):
            for col in range(1, self.cols + 1):
                state = (row, col)
                if state == self.start_state:
                    cell_type = "START"
                elif state == self.goal_state:
                    cell_type = "GOAL"
                elif state == self.jump_in:
                    cell_type = "JUMP_IN"
                elif state == self.jump_out:
                    cell_type = "JUMP_OUT"
                elif state in self.obstacles:
                    cell_type = "OBSTACLE"
                else:
                    cell_type = "FREE"
                rows.append({"row": row, "column": col, "cell_type": cell_type})
        return rows
