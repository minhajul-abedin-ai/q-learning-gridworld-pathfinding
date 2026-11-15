"""Regression tests for the Q-learning update and action selection."""

import numpy as np
import pytest

from src.q_learning_agent import QLearningAgent


@pytest.mark.parametrize("parameter", ["alpha", "gamma", "epsilon"])
@pytest.mark.parametrize("out_of_range", [-0.01, 1.01])
def test_rejects_out_of_range_hyperparameters(parameter, out_of_range):
    with pytest.raises(ValueError, match=parameter):
        QLearningAgent(**{parameter: out_of_range})


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_accepts_hyperparameter_boundaries(boundary):
    agent = QLearningAgent(alpha=boundary, gamma=boundary, epsilon=boundary)
    assert (agent.alpha, agent.gamma, agent.epsilon) == (boundary,) * 3


def test_greedy_action_selects_highest_q_value_without_exploration():
    agent = QLearningAgent(epsilon=0.0)
    agent.q_table[1, 0] = [0.0, 4.0, 2.0, 3.0]
    assert agent.select_action((2, 1)) == 1


def test_exploration_is_reproducible_for_a_fixed_seed():
    first = QLearningAgent(epsilon=1.0, seed=123)
    second = QLearningAgent(epsilon=1.0, seed=123)
    actions_one = [first.select_action((2, 1)) for _ in range(30)]
    actions_two = [second.select_action((2, 1)) for _ in range(30)]
    assert actions_one == actions_two
    assert all(0 <= action < 4 for action in actions_one)


def test_nonterminal_update_bootstraps_from_best_next_action():
    agent = QLearningAgent(alpha=0.25, gamma=0.5)
    agent.q_table[1, 1] = [1.0, 8.0, 3.0, -2.0]
    agent.update((2, 1), action_index=2, reward=2.0, next_state=(2, 2), done=False)
    # target=2 + 0.5*8 = 6; updated Q = 0 + 0.25*6 = 1.5
    assert agent.q_table[1, 0, 2] == pytest.approx(1.5)


def test_terminal_update_never_bootstraps_from_next_state():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    agent.q_table[4, 3, 2] = 4.0
    agent.q_table[4, 4] = [100.0, 0.0, 0.0, 0.0]
    agent.update((5, 4), action_index=2, reward=10.0, next_state=(5, 5), done=True)
    # terminal target=10, so updated Q = 4 + 0.5*(10 - 4) = 7
    assert agent.q_table[4, 3, 2] == pytest.approx(7.0)


def test_state_values_and_greedy_policy_use_action_axis():
    agent = QLearningAgent()
    agent.q_table[0, 0] = [0.0, 2.0, 7.0, 3.0]
    values = agent.state_values()
    policy = agent.greedy_policy()
    assert values.shape == (5, 5)
    assert policy.shape == (5, 5)
    assert values[0, 0] == pytest.approx(7.0)
    assert policy[0, 0] == 2
    np.testing.assert_array_equal(values[1:, :], np.zeros((4, 5)))
