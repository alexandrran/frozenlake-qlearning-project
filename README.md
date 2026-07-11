# FrozenLake Q-Learning

Tabular Q-learning agent for the Gymnasium `FrozenLake-v1` environment.
The agent learns through trial and error to reach the goal while avoiding holes.

## Features

- Supports the standard `4x4` and `8x8` FrozenLake maps.
- Supports slippery and deterministic ice modes.
- Uses epsilon-greedy exploration and the Q-learning update rule.
- Saves a training curve, a learned-policy map, and a numerical Q-table heatmap.
- Evaluates the agent before and after training.
- Uses fixed random seeds for reproducible experiments.

## Installation

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Configuration

Open `frozen_lake_q.py` and adjust these settings:

```python
MAP_NAME = "8x8"       # "4x4" or "8x8"
IS_SLIPPERY = True     # True for stochastic movement
```

## Run

Train the agent by enabling the training call at the bottom of the file:

```python
run(40000)
```

Run a visual demonstration of a saved model:

```python
run(5, is_training=False, render=True)
```

Create the policy and Q-table images from a saved model without retraining:

```python
visualize_saved_q_table()
```

Only one of these calls should be active at a time.

## Results

Training artifacts are stored by map size:

```text
results/
+-- 4x4/
|   +-- q_table.pkl
|   +-- training_progress.png
|   +-- policy.png
|   `-- q_table.png
`-- 8x8/
    +-- q_table.pkl
    +-- training_progress.png
    +-- policy.png
    `-- q_table.png
```

The `assets/maps/` folder contains screenshots of the standard 4x4 and 8x8 FrozenLake maps.

## Q-Learning Update

```text
Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max Q(s', a') - Q(s, a)]
```

The Q-table has one row per state and one column per action. Therefore, it has shape `16 x 4` for the 4x4 map and `64 x 4` for the 8x8 map.
