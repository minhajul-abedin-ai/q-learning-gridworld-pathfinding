# Data directory

This reinforcement-learning project does **not** rely on an external training dataset.
The agent learns by interacting directly with the Grid World environment.

Running `python train.py` generates the CSV files in this folder:

- `environment_layout.csv` — coordinates and cell types in the 5×5 world.
- `training_log_*.csv` — episode-level rewards, steps and 30-episode moving averages.
- `epsilon_experiment_summary.csv` — comparison of exploration rates.
- `learning_rate_experiment_summary.csv` — comparison of learning rates.
- `final_q_values.csv` — learned Q-values for North, South, East and West at every state.

These generated logs serve as the project's reproducible experimental dataset.
