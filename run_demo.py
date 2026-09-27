"""Small console demo showing the greedy route learned by the final agent."""

from src.experiments import train_agent
from src.grid_environment import GridWorld


def main():
    result = train_agent(alpha=0.2, gamma=0.9, epsilon=0.1, seed=42)
    agent = result.agent
    env = GridWorld()
    state = env.reset()

    route = [state]
    total_reward = 0.0
    seen_steps = 0

    while state != env.goal_state and seen_steps < 50:
        row, col = state[0] - 1, state[1] - 1
        action_index = int(agent.q_table[row, col].argmax())
        action = env.ACTIONS[action_index]
        transition = env.step(action)
        state = transition.next_state
        total_reward += transition.reward
        route.append(state)
        seen_steps += 1
        if transition.done:
            break

    print("Greedy route:", " -> ".join(map(str, route)))
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()
