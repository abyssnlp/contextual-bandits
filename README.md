# Contextual Bandits for Marketing Campaigns

This repository provides a framework for simulating and comparing contextual bandit algorithms on synthetic marketing data. The main algorithms implemented are LinUCB and a feed-forward Neural Bandit. The simulation environment is designed to evaluate online learning strategies for campaign selection.

## Algorithms

### LinUCB
LinUCB is a linear contextual bandit algorithm. It models the expected reward for each arm as a linear function of the context features. At each step, it selects the arm with the highest upper confidence bound, balancing exploration and exploitation. LinUCB is efficient and works well when the true reward function is approximately linear in the features.

### Neural Bandit
The Neural Bandit algorithm uses a feed-forward neural network to estimate the expected reward for each arm given the context. It can capture non-linear relationships between context and reward. The neural network is trained online as new data arrives, allowing the algorithm to adapt to complex reward structures.

## Running the Simulation

The main simulation script is `examples/run_simulation.py`. It generates synthetic marketing data, runs the bandit algorithms in an online environment, and plots the results.

### Requirements
- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn

Install dependencies using pip or poetry as appropriate for your environment.

### Usage

From the project root directory, run:

```
python examples/run_simulation.py
```

This will:
- Generate synthetic user and campaign data
- Run LinUCB, Neural Bandit, and a random baseline in an online simulation
- Plot cumulative rewards, regret, and other metrics
- Print a summary of results

You can modify the script to change the number of campaigns, features, or samples as needed.

## File Structure
- `contextual_bandits/algorithms/linucb.py`: LinUCB implementation
- `contextual_bandits/algorithms/neural_bandit.py`: Neural Bandit implementation
- `contextual_bandits/utils/preprocessing.py`: Data preprocessing utilities
- `examples/generate_data.py`: Synthetic data generation
- `examples/run_simulation.py`: Main simulation script

## License
This project is for research and educational purposes.