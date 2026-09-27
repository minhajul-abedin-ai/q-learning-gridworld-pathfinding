"""Create a GIF of the greedy route learned by the Q-learning agent."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from src.experiments import train_agent
from src.grid_environment import GridWorld

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "agent_path_animation.gif"


def learned_route():
    result = train_agent(alpha=0.2, gamma=0.9, epsilon=0.1, seed=42)
    agent = result.agent
    env = GridWorld()
    state = env.reset()
    route = [state]

    for _ in range(50):
        row, col = state[0] - 1, state[1] - 1
        action_index = int(agent.q_table[row, col].argmax())
        transition = env.step(env.ACTIONS[action_index])
        state = transition.next_state
        route.append(state)
        if transition.done:
            break
    return route


def draw_grid(ax, env, agent_state):
    ax.clear()
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(5.5, 0.5)
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.grid(True, linewidth=1)
    ax.set_aspect("equal")
    ax.set_title("Q-Learning Agent: Learned Greedy Route")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    for row, col in env.obstacles:
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, alpha=0.45))
        ax.text(col, row, "X", ha="center", va="center", fontweight="bold")

    ax.text(env.start_state[1], env.start_state[0], "START", ha="center", va="center", fontsize=8)
    ax.text(env.goal_state[1], env.goal_state[0], "GOAL", ha="center", va="center", fontsize=8)
    ax.text(env.jump_in[1], env.jump_in[0], "JUMP", ha="center", va="center", fontsize=8)
    ax.text(env.jump_out[1], env.jump_out[0], "OUT", ha="center", va="center", fontsize=8)

    ax.scatter([agent_state[1]], [agent_state[0]], s=300, marker="o", zorder=5)
    ax.text(agent_state[1], agent_state[0], "A", ha="center", va="center", fontweight="bold", zorder=6)


def main():
    env = GridWorld()
    route = learned_route()
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        draw_grid(ax, env, route[frame])
        ax.text(0.02, 1.02, f"Step {frame}: {route[frame]}", transform=ax.transAxes)

    animation = FuncAnimation(fig, update, frames=len(route), interval=900, repeat=True)
    animation.save(OUTPUT, writer=PillowWriter(fps=1))
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
