"""Tests for the Q-learning agent."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from maze import Maze
from qlearning import QLearningAgent


class TestQLearningAgent(unittest.TestCase):
    """Test cases for the QLearningAgent class."""

    def test_q_table_shape(self):
        maze = Maze()
        agent = QLearningAgent(maze.get_num_states(), maze.num_actions)
        self.assertEqual(agent.q_table.shape, (maze.get_num_states(), maze.num_actions))

    def test_update_changes_q_value(self):
        maze = Maze([[2, 3]])
        agent = QLearningAgent(
            maze.get_num_states(),
            maze.num_actions,
            learning_rate=1.0,
            discount_factor=0.0,
        )

        state_idx = maze.get_state_index((0, 0))
        next_state, reward, done = maze.step(3)
        next_state_idx = maze.get_state_index(next_state)
        agent.update(state_idx, 3, reward, next_state_idx, done)

        self.assertEqual(agent.q_table[state_idx, 3], 100)

    def test_epsilon_decay_respects_minimum(self):
        maze = Maze()
        agent = QLearningAgent(
            maze.get_num_states(),
            maze.num_actions,
            epsilon=0.02,
            epsilon_decay=0.1,
            epsilon_min=0.01,
        )

        agent.decay_epsilon()
        self.assertEqual(agent.epsilon, 0.01)

    def test_best_path_starts_at_maze_start(self):
        maze = Maze()
        agent = QLearningAgent(maze.get_num_states(), maze.num_actions)
        path = agent.get_best_path(maze, max_steps=5)
        self.assertEqual(path[0], maze.start_pos)


if __name__ == '__main__':
    unittest.main()
