# Q-Learning Maze Solver

Reinforcement learning project comparing four maze-solving algorithms: Q-Learning, SARSA, Boltzmann Q-Learning, and Ant Colony Optimization.

## Algorithms

| Algorithm | Type | Exploration Strategy |
|-----------|------|---------------------|
| Q-Learning | Off-policy TD control | ε-greedy |
| SARSA | On-policy TD control | ε-greedy |
| Boltzmann Q-Learning | Off-policy TD control | Softmax (temperature-based) |
| Ant Colony Optimization | Swarm intelligence | Pheromone-guided probabilistic |

## Features

- Grid-based maze environment with configurable walls, start, and goal positions
- Tabular Q-learning agent with ε-greedy exploration
- SARSA agent for on-policy comparison
- Boltzmann exploration agent with temperature parameter control
- Ant Colony Optimization using pheromone trails and heuristic guidance
- Matplotlib animation for path visualization
- Sensitivity analysis across hyperparameters (γ, τ, ρ)
- Generalization testing on different maze topologies

## Methodology

### Reinforcement Learning Agents

**Q-Learning** updates Q-values using the maximum expected future reward:
```
Q(s,a) += α * (r + γ * max Q(s',a') - Q(s,a))
```

**SARSA** updates Q-values using the actual next action taken:
```
Q(s,a) += α * (r + γ * Q(s',a') - Q(s,a))
```

**Boltzmann Q-Learning** replaces ε-greedy with softmax action selection:
```
P(a) = exp(Q(s,a)/τ) / Σ exp(Q(s,a')/τ)
```

**Ant Colony Optimization** uses pheromone deposits proportional to path quality:
```
τ(i,j) += Q/path_length for successful paths
τ(i,j) *= (1 - ρ) for evaporation
```

### Sensitivity Analysis

Tested parameter sweeps:
- **Discount factor (γ):** [0.1, 0.5, 0.9, 0.99]
- **Boltzmann temperature (τ):** [0.1, 0.5, 1.0, 2.0, 5.0]
- **ACO evaporation rate (ρ):** [0.05, 0.1, 0.3, 0.5]

### Generalization

Both 5×5 and 3×7 bridge maze topologies used to test algorithm adaptability.

## Project Structure

```
Q-learning-maze-solver/
├── Q_Learning_Maze_Solver.ipynb   # Main notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

```
numpy
matplotlib
seaborn
```

## Usage

```bash
pip install -r requirements.txt
jupyter notebook Q_Learning_Maze_Solver.ipynb
```

## License

MIT
