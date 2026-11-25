"""Regression checks for terminal-state Q-learning targets."""

import pytest

from src.q_learning_agent import QLearningAgent


def test_terminal_update_does_not_bootstrap_from_next_state():
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0, seed=42)
    agent.q_table[4, 4, :] = 100.0  # Artificially high terminal-state estimates.
    agent.q_table[4, 3, 2] = 2.0

    agent.update((5, 4), 2, 10.0, (5, 5), done=True)

    assert agent.q_table[4, 3, 2] == pytest.approx(6.0)


def test_nonterminal_update_bootstraps_from_best_next_action():
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0, seed=42)
    agent.q_table[1, 1, :] = [1.0, 4.0, 2.0, 3.0]
    agent.q_table[1, 0, 2] = 2.0

    agent.update((2, 1), 2, -1.0, (2, 2), done=False)

    assert agent.q_table[1, 0, 2] == pytest.approx(2.3)


def test_zero_learning_rate_leaves_q_value_unchanged():
    agent = QLearningAgent(alpha=0.0, gamma=0.9, epsilon=0, seed=42)
    agent.q_table[4, 3, 2] = 7.0

    agent.update((5, 4), 2, 10.0, (5, 5), done=True)

    assert agent.q_table[4, 3, 2] == pytest.approx(7.0)
