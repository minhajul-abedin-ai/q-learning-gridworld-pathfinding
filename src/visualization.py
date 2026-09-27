"""Plotting helpers for the Grid World experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .grid_environment import GridWorld


def plot_experiment_curves(results, parameter_name: str, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for value, result in results.items():
        ax.plot(result.log["episode"], result.log["cumulative_reward"], label=f"{parameter_name}={value}")
    ax.axhline(10.0, linestyle="--", linewidth=1, label="Early-stop threshold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title(f"Cumulative Reward by {parameter_name}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_steps(log, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(log["episode"], log["steps"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Steps per Episode")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_state_values(agent, output_path: str | Path) -> None:
    env = GridWorld()
    values = agent.state_values().copy()

    for row, col in env.obstacles:
        values[row - 1, col - 1] = np.nan

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(values, origin="upper")
    fig.colorbar(image, ax=ax, label="max Q(s,a)")

    for row in range(1, env.rows + 1):
        for col in range(1, env.cols + 1):
            state = (row, col)
            if state in env.obstacles:
                label = "OBSTACLE"
            elif state == env.start_state:
                label = f"START\n{values[row-1, col-1]:.2f}"
            elif state == env.goal_state:
                label = "GOAL\n+10"
            elif state == env.jump_in:
                label = "JUMP IN"
            elif state == env.jump_out:
                label = f"JUMP OUT\n{values[row-1, col-1]:.2f}"
            else:
                label = f"{values[row-1, col-1]:.2f}"
            ax.text(col - 1, row - 1, label, ha="center", va="center", fontsize=8)

    ax.set_xticks(range(5), labels=range(1, 6))
    ax.set_yticks(range(5), labels=range(1, 6))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Learned State Values")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
