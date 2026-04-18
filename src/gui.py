"""GUI visualization for the Q-learning maze solver."""

from __future__ import annotations

import numpy as np
import pygame

from maze import Maze
from qlearning import QLearningAgent


class MazeGUI:
    """Visualize the maze and the agent's learning process."""

    COLORS = {
        "background": (248, 245, 238),
        "panel": (34, 40, 49),
        "panel_text": (244, 244, 244),
        "wall": (39, 55, 77),
        "empty": (232, 236, 241),
        "start": (125, 181, 130),
        "goal": (224, 122, 95),
        "agent": (61, 133, 198),
        "agent_outline": (24, 79, 120),
        "grid": (200, 205, 214),
        "policy": (93, 154, 186),
        "muted": (173, 181, 189),
    }

    def __init__(self, maze, cell_size=80, fps=10):
        self.maze = maze
        self.cell_size = cell_size
        self.fps = fps
        self.width = maze.cols * cell_size
        self.height = maze.rows * cell_size + 120

        pygame.init()
        pygame.display.set_caption("Q-Learning Maze Solver")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 22)

        self.running = True
        self.mode = "watch"
        self.agent = None
        self.episode = 0
        self.step_count = 0
        self.rewards_history = []
        self.current_episode_reward = 0
        self.state = self.maze.reset()
        self.state_idx = self.maze.get_state_index(self.state)

    def reset_episode_state(self):
        self.state = self.maze.reset()
        self.state_idx = self.maze.get_state_index(self.state)
        self.step_count = 0

    def draw_grid(self):
        for row in range(self.maze.rows):
            for col in range(self.maze.cols):
                x = col * self.cell_size
                y = row * self.cell_size

                if self.maze.maze[row, col] == 1:
                    color = self.COLORS["wall"]
                elif self.maze.maze[row, col] == 2:
                    color = self.COLORS["start"]
                elif self.maze.maze[row, col] == 3:
                    color = self.COLORS["goal"]
                else:
                    color = self.COLORS["empty"]

                pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(
                    self.screen,
                    self.COLORS["grid"],
                    (x, y, self.cell_size, self.cell_size),
                    1,
                )

    def draw_agent(self, position):
        row, col = position
        center_x = col * self.cell_size + self.cell_size // 2
        center_y = row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3
        pygame.draw.circle(self.screen, self.COLORS["agent"], (center_x, center_y), radius)
        pygame.draw.circle(
            self.screen,
            self.COLORS["agent_outline"],
            (center_x, center_y),
            radius,
            2,
        )

    def draw_q_values(self):
        if self.agent is None or self.mode != "watch":
            return

        action_offsets = {
            0: (0, -10),
            1: (0, 10),
            2: (-10, 0),
            3: (10, 0),
        }

        for row in range(self.maze.rows):
            for col in range(self.maze.cols):
                if self.maze.maze[row, col] == 1:
                    continue

                state_idx = self.maze.get_state_index((row, col))
                q_values = self.agent.q_table[state_idx]
                best_action = int(np.argmax(q_values))
                best_q = q_values[best_action]

                if abs(best_q) <= 0.1:
                    continue

                center_x = col * self.cell_size + self.cell_size // 2
                center_y = row * self.cell_size + self.cell_size // 2
                dx, dy = action_offsets[best_action]
                pygame.draw.line(
                    self.screen,
                    self.COLORS["policy"],
                    (center_x, center_y),
                    (center_x + dx, center_y + dy),
                    3,
                )

    def draw_info_panel(self):
        panel_y = self.maze.rows * self.cell_size
        pygame.draw.rect(self.screen, self.COLORS["panel"], (0, panel_y, self.width, 120))

        lines = [
            f"Mode: {self.mode.upper()}",
            f"Episode: {self.episode}",
            f"Step: {self.step_count}",
        ]

        if self.agent is not None:
            lines.append(f"Epsilon: {self.agent.epsilon:.4f}")
        if self.rewards_history:
            avg_reward = np.mean(self.rewards_history[-100:])
            lines.append(f"Avg Reward (last 100): {avg_reward:.1f}")

        for index, text in enumerate(lines):
            surface = self.font.render(text, True, self.COLORS["panel_text"])
            self.screen.blit(surface, (16 + (index // 3) * 220, panel_y + 16 + (index % 3) * 28))

        controls = "T train | S solve | W watch | R reset | Q quit"
        surface = self.small_font.render(controls, True, self.COLORS["muted"])
        self.screen.blit(surface, (16, panel_y + 94))

    def render(self):
        self.screen.fill(self.COLORS["background"])
        self.draw_grid()
        self.draw_q_values()
        self.draw_agent(self.maze.current_pos)
        self.draw_info_panel()
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_t:
                    self.mode = "train"
                    self.reset_training()
                elif event.key == pygame.K_s:
                    self.mode = "solve"
                    self.reset_solve()
                elif event.key == pygame.K_w:
                    self.mode = "watch"
                    self.reset_watch()
                elif event.key == pygame.K_r:
                    self.reset_all()

    def reset_all(self):
        self.agent = None
        self.episode = 0
        self.rewards_history = []
        self.current_episode_reward = 0
        self.reset_episode_state()

    def reset_training(self):
        self.agent = QLearningAgent(self.maze.get_num_states(), self.maze.num_actions)
        self.episode = 0
        self.rewards_history = []
        self.current_episode_reward = 0
        self.reset_episode_state()

    def quick_train(self, episodes=500):
        self.agent = QLearningAgent(self.maze.get_num_states(), self.maze.num_actions)
        self.rewards_history = self.agent.train(
            self.maze,
            num_episodes=episodes,
            max_steps_per_episode=max(self.maze.get_num_states() * 6, 50),
            verbose_interval=0,
        )
        self.reset_episode_state()

    def reset_solve(self):
        if self.agent is None:
            self.quick_train()
        self.reset_episode_state()

    def reset_watch(self):
        if self.agent is None:
            self.quick_train(episodes=250)
        self.reset_episode_state()

    def train_step(self):
        if self.agent is None:
            self.reset_training()
            return

        action = self.agent.get_action(self.state_idx)
        next_state, reward, done = self.maze.step(action)
        next_state_idx = self.maze.get_state_index(next_state)
        self.agent.update(self.state_idx, action, reward, next_state_idx, done)

        self.state = next_state
        self.state_idx = next_state_idx
        self.current_episode_reward += reward
        self.step_count += 1

        if done:
            self.agent.decay_epsilon()
            self.rewards_history.append(self.current_episode_reward)
            self.episode += 1
            self.current_episode_reward = 0
            self.reset_episode_state()

    def solve_step(self):
        if self.agent is None:
            self.quick_train()

        action = int(np.argmax(self.agent.q_table[self.state_idx]))
        next_state, _, done = self.maze.step(action)
        self.state = next_state
        self.state_idx = self.maze.get_state_index(next_state)
        self.step_count += 1

        if done:
            pygame.time.wait(400)
            self.reset_episode_state()

    def run(self):
        self.reset_all()

        while self.running:
            self.handle_events()

            if self.mode == "train":
                self.train_step()
            elif self.mode == "solve":
                self.solve_step()

            self.render()
            self.clock.tick(self.fps)

        pygame.quit()


def main():
    print("=" * 50)
    print("Q-Learning Maze Solver - GUI Visualization")
    print("=" * 50)
    print("\nControls:")
    print("  [T] Train in real time")
    print("  [S] Solve with the learned policy")
    print("  [W] Watch the current policy")
    print("  [R] Reset")
    print("  [Q] Quit")

    gui = MazeGUI(Maze(), cell_size=80, fps=15)
    gui.run()
    print("\nGUI closed.")


if __name__ == "__main__":
    main()
