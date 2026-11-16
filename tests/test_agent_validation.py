"""Regression tests for invalid Q-learning hyperparameters."""

import math

import pytest

from src.q_learning_agent import QLearningAgent


@pytest.mark.parametrize("parameter", ["alpha", "gamma", "epsilon"])
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_reject_nonfinite_hyperparameters(parameter, invalid):
    with pytest.raises(ValueError, match=parameter):
        QLearningAgent(**{parameter: invalid})


@pytest.mark.parametrize("parameter", ["alpha", "gamma", "epsilon"])
def test_allow_finite_boundary_values(parameter):
    for value in (0.0, 1.0):
        agent = QLearningAgent(**{parameter: value})
        assert getattr(agent, parameter) == value
