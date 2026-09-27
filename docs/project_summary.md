# Project summary

## Problem

Train an autonomous agent to navigate a deterministic 5×5 Grid World from a fixed start state to a terminal goal while avoiding obstacles and learning to exploit a rewarded jump state.

## Environment

- Start: `(2, 1)`
- Goal: `(5, 5)` with reward `+10`
- Jump-in: `(2, 4)`
- Jump-out: `(4, 4)` with reward `+5`
- Obstacles: `(3,3)`, `(3,4)`, `(3,5)`, `(4,3)`
- Normal move or invalid move: `-1`
- Actions: North, South, East, West

## Learning algorithm

The project uses off-policy tabular Q-learning:

`Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]`

The behaviour policy is epsilon-greedy. Training is capped at 100 episodes and uses a 30-episode moving average for early stopping when the average reward is greater than 10.

## Experiments

The repository reproduces two experiment groups:

1. Epsilon comparison: `0.1`, `0.3`, `0.5`, `0.9`.
2. Learning-rate comparison: `0.2`, `0.4`, `0.8`, `1.0`.

A fixed seed is used to make the included generated outputs reproducible.

## Important note

The source code in this repository is a clean reconstruction of the implementation described in the original academic assignment. The generated CSV files and figures are produced by this repository's code, so their exact numeric results may differ from figures that appeared in the original report if the original run used a different random state or implementation detail.
