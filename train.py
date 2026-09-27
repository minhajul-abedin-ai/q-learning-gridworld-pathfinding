"""Run all experiments and export the project dataset/results."""

from pathlib import Path

import pandas as pd

from src.experiments import run_epsilon_experiment, run_learning_rate_experiment, train_agent
from src.grid_environment import GridWorld
from src.visualization import plot_experiment_curves, plot_state_values, plot_steps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def export_environment_layout():
    env = GridWorld()
    pd.DataFrame(env.layout_rows()).to_csv(DATA / "environment_layout.csv", index=False)


def export_q_values(agent):
    rows = []
    for row in range(1, 6):
        for col in range(1, 6):
            values = agent.q_table[row - 1, col - 1]
            rows.append(
                {
                    "row": row,
                    "column": col,
                    "Q_N": values[0],
                    "Q_S": values[1],
                    "Q_E": values[2],
                    "Q_W": values[3],
                    "state_value": values.max(),
                }
            )
    pd.DataFrame(rows).to_csv(DATA / "final_q_values.csv", index=False)


def main():
    export_environment_layout()

    epsilon_results, epsilon_summary = run_epsilon_experiment()
    epsilon_summary.to_csv(DATA / "epsilon_experiment_summary.csv", index=False)
    for epsilon, result in epsilon_results.items():
        slug = str(epsilon).replace(".", "_")
        result.log.to_csv(DATA / f"training_log_epsilon_{slug}.csv", index=False)
    plot_experiment_curves(epsilon_results, "epsilon", RESULTS / "epsilon_comparison.png")

    lr_results, lr_summary = run_learning_rate_experiment()
    lr_summary.to_csv(DATA / "learning_rate_experiment_summary.csv", index=False)
    for alpha, result in lr_results.items():
        slug = str(alpha).replace(".", "_")
        result.log.to_csv(DATA / f"training_log_alpha_{slug}.csv", index=False)
    plot_experiment_curves(lr_results, "alpha", RESULTS / "learning_rate_comparison.png")

    final_result = train_agent(alpha=0.2, gamma=0.9, epsilon=0.1, seed=42)
    final_result.log.to_csv(DATA / "final_agent_training_log.csv", index=False)
    export_q_values(final_result.agent)
    plot_steps(final_result.log, RESULTS / "steps_per_episode.png")
    plot_state_values(final_result.agent, RESULTS / "learned_state_values.png")

    print("Project outputs generated successfully.")
    print(f"Final run episodes: {len(final_result.log)}")
    print(f"Final 30-episode moving average: {final_result.final_moving_average:.3f}")


if __name__ == "__main__":
    main()
