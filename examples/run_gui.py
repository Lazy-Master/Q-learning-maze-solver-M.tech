"""
Example: Run the Q-Learning Maze Solver GUI

This script launches the interactive GUI visualization where you can:
- Watch the agent learn in real-time (press T)
- Watch the trained agent solve the maze (press S)
- Observe the current state (press W)
- Reset everything (press R)
- Quit (press Q)

Requirements:
    pip install pygame

Usage:
    python examples/run_gui.py
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gui import main

if __name__ == "__main__":
    main()
