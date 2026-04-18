"""Q-learning agent for maze solving."""

from __future__ import annotations

import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent for grid mazes."""

    ACTION_NAMES = ["Up", "Down", "Left", "Right"]

    def __init__(
        self,
        num_states,
        num_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions), dtype=float)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_greedy_action(self, state):
        """Return the best known action for a state."""
        return int(np.argmax(self.q_table[state]))

    def get_action(self, state):
        """Select an action using an epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        return self.get_greedy_action(state)

    def update(self, state, action, reward, next_state, done):
        """Update the Q-value for a state/action pair."""
        max_next_q = 0 if done else np.max(self.q_table[next_state])
        current_q = self.q_table[state, action]
        target = reward + self.discount_factor * max_next_q
        self.q_table[state, action] = current_q + self.learning_rate * (target - current_q)

    def decay_epsilon(self):
        """Decay the exploration rate without dropping below the configured floor."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(
        self,
        env,
        num_episodes=1000,
        render=False,
        max_steps_per_episode=None,
        verbose_interval=100,
    ):
        """Train the agent on the provided maze environment."""
        if max_steps_per_episode is None:
            max_steps_per_episode = max(env.get_num_states() * 4, 25)

        rewards_per_episode = []

        for episode in range(num_episodes):
            state = env.reset()
            state_idx = env.get_state_index(state)
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < max_steps_per_episode:
                action = self.get_action(state_idx)
                next_state, reward, done = env.step(action)
                next_state_idx = env.get_state_index(next_state)
                self.update(state_idx, action, reward, next_state_idx, done)
                state_idx = next_state_idx
                total_reward += reward
                steps += 1

                if render and verbose_interval and episode % verbose_interval == 0:
                    env.render()

            self.decay_epsilon()
            rewards_per_episode.append(total_reward)

            if verbose_interval and episode % verbose_interval == 0:
                print(
                    f"Episode {episode:>4} | Reward: {total_reward:>4} | "
                    f"Steps: {steps:>3} | Epsilon: {self.epsilon:.4f}"
                )

        return rewards_per_episode

    def get_best_path(self, env, max_steps=100):
        """Return the greedy path learned by the agent."""
        state = env.reset()
        state_idx = env.get_state_index(state)
        path = [state]
        done = False

        for _ in range(max_steps):
            if done:
                break

            action = self.get_greedy_action(state_idx)
            next_state, _, done = env.step(action)
            next_state_idx = env.get_state_index(next_state)
            path.append(next_state)

            if next_state == state and not done:
                break

            state = next_state
            state_idx = next_state_idx

        return path

    def summarize_q_table(self, env, threshold=0.01):
        """Return a readable summary of learned Q-values above a threshold."""
        lines = []
        for state, action in zip(*np.where(np.abs(self.q_table) > threshold)):
            row = state // env.cols
            col = state % env.cols
            q_value = self.q_table[state, action]
            lines.append(
                f"State ({row}, {col}), Action {self.ACTION_NAMES[action]}: Q={q_value:.2f}"
            )
        return lines


def main():
    """Run a simple training session on the default maze."""
    from maze import Maze

    print("=" * 50)
    print("Q-Learning Maze Solver")
    print("=" * 50)

    env = Maze()
    print("\nInitial Maze:")
    print(env)

    agent = QLearningAgent(env.get_num_states(), env.num_actions)

    print("\nTraining the agent...")
    rewards = agent.train(env, num_episodes=1000, render=False)

    print("\nTraining completed!")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")

    path = agent.get_best_path(env)
    print(f"\nGreedy path length: {len(path) - 1} steps")
    print("Path:", path)

    print("\nFinal maze state:")
    env.reset()
    for position in path[1:]:
        env.current_pos = position
    env.render()

    print("Q-table summary:")
    for line in agent.summarize_q_table(env):
        print(line)


if __name__ == "__main__":
    main()
