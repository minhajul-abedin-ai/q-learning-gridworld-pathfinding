"""Training and experiment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from .grid_environment import GridWorld
from .q_learning_agent import QLearningAgent


@dataclass
class TrainingResult:
    agent: QLearningAgent
    log: pd.DataFrame
    converged_episode: int | None
    final_moving_average: float


def train_agent(
    alpha: float = 0.2,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    seed: int = 42,
    max_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    moving_average_window: int = 30,
    early_stop_threshold: float = 10.0,
) -> TrainingResult:
    """Train one Q-learning agent and return episode-level logs."""
    env = GridWorld()
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)

    records: List[dict] = []
    rewards: List[float] = []
    converged_episode = None

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        reached_goal = False

        for step_number in range(1, max_steps_per_episode + 1):
            action_index = agent.select_action(state)
            action = env.ACTIONS[action_index]
            transition = env.step(action)

            agent.update(
                state=state,
                action_index=action_index,
                reward=transition.reward,
                next_state=transition.next_state,
                done=transition.done,
            )

            total_reward += transition.reward
            state = transition.next_state

            if transition.done:
                reached_goal = True
                break

        rewards.append(total_reward)
        moving_average = (
            float(np.mean(rewards[-moving_average_window:]))
            if len(rewards) >= moving_average_window
            else np.nan
        )

        records.append(
            {
                "episode": episode,
                "cumulative_reward": total_reward,
                "steps": step_number,
                "moving_average_30": moving_average,
                "reached_goal": reached_goal,
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "seed": seed,
            }
        )

        if len(rewards) >= moving_average_window and moving_average > early_stop_threshold:
            converged_episode = episode
            break

    log = pd.DataFrame(records)
    final_ma = float(log["moving_average_30"].dropna().iloc[-1]) if log["moving_average_30"].notna().any() else float("nan")
    return TrainingResult(agent, log, converged_episode, final_ma)


def run_epsilon_experiment(
    epsilons: Iterable[float] = (0.1, 0.3, 0.5, 0.9),
    alpha: float = 0.2,
    gamma: float = 0.9,
    seed: int = 42,
):
    results = {}
    summary = []

    for epsilon in epsilons:
        result = train_agent(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
        results[epsilon] = result
        summary.append(
            {
                "epsilon": epsilon,
                "converged_episode": result.converged_episode if result.converged_episode is not None else "No stop",
                "episodes_run": len(result.log),
                "final_30_episode_average": result.final_moving_average,
            }
        )

    return results, pd.DataFrame(summary)


def run_learning_rate_experiment(
    learning_rates: Iterable[float] = (0.2, 0.4, 0.8, 1.0),
    epsilon: float = 0.1,
    gamma: float = 0.9,
    seed: int = 42,
):
    results = {}
    summary = []

    for alpha in learning_rates:
        result = train_agent(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
        results[alpha] = result
        summary.append(
            {
                "alpha": alpha,
                "converged_episode": result.converged_episode if result.converged_episode is not None else "No stop",
                "episodes_run": len(result.log),
                "final_30_episode_average": result.final_moving_average,
            }
        )

    return results, pd.DataFrame(summary)
