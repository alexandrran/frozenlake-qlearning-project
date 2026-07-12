import argparse
import gymnasium as gym
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


MAP_NAME = "8x8"
IS_SLIPPERY = True
RESULTS_DIR = Path("results")
TRAINING_SEED = 42
EVALUATION_SEED = 12345
EPSILON_DECAY_EPISODES = {
    "4x4": 5000,
    "8x8": 25000,
}
MAP_LAYOUTS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


def run_name():
    return MAP_NAME


def artifact_path(filename):
    output_dir = RESULTS_DIR / run_name()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def select_best_action(q, state, rng):
    state_actions = q[state, :]
    best_actions = np.flatnonzero(state_actions == np.max(state_actions))
    return int(rng.choice(best_actions))


def update_q_value(q, state, action, reward, new_state, done, alpha, gamma):
    future_q = 0 if done else np.max(q[new_state, :])
    q[state, action] += alpha * (
        reward + gamma * future_q - q[state, action]
    )


def evaluate_agent(q, episodes=1000, seed=EVALUATION_SEED):
    env = gym.make(
        'FrozenLake-v1',
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
    )
    wins = 0
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)

    for episode in range(episodes):
        state = env.reset(seed=seed if episode == 0 else None)[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if q is None:
                action = env.action_space.sample()
            else:
                action = select_best_action(q, state, rng)

            state, reward, terminated, truncated, _ = env.step(action)

        wins += int(reward == 1)

    env.close()
    success_rate = wins / episodes * 100
    return wins, success_rate


def save_q_table_visualizations(q):
    map_layout = MAP_LAYOUTS[MAP_NAME]
    rows = len(map_layout)
    cols = len(map_layout[0])
    state_values = np.max(q, axis=1).reshape(rows, cols)
    best_actions = np.argmax(q, axis=1).reshape(rows, cols)
    action_symbols = ['<', 'v', '>', '^']

    fig, ax = plt.subplots(figsize=(max(6, cols), max(5, rows * 0.85)))
    image = ax.imshow(state_values, cmap='YlGnBu')

    for row, map_row in enumerate(map_layout):
        for col, tile in enumerate(map_row):
            value = state_values[row, col]
            if tile in ('H', 'G'):
                label = tile
            elif value == 0:
                label = 'S' if tile == 'S' else '-'
            else:
                action = action_symbols[best_actions[row, col]]
                label = f'S\n{action}' if tile == 'S' else action

            text_color = 'white' if value > state_values.max() * 0.55 else 'black'
            ax.text(col, row, label, ha='center', va='center', color=text_color, fontsize=12)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(f'FrozenLake {MAP_NAME} learned policy from the Q-table')
    fig.colorbar(image, ax=ax, label='Maximum Q-value')
    fig.tight_layout()
    fig.savefig(artifact_path('policy.png'), dpi=160)
    plt.close(fig)

    # Full Q-table: one row per state and one column per action.
    n_states, n_actions = q.shape
    figure_height = max(6, n_states * 0.24)
    fig, ax = plt.subplots(figsize=(8, figure_height))
    image = ax.imshow(q, aspect='auto', cmap='viridis')

    ax.set_xticks(range(n_actions))
    ax.set_xticklabels(['Left', 'Down', 'Right', 'Up'])
    ax.set_yticks(range(n_states))
    ax.tick_params(axis='y', labelsize=7)
    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    ax.set_title(f'FrozenLake {MAP_NAME} Q-table values Q(state, action)')

    value_min = np.min(q)
    value_max = np.max(q)
    midpoint = (value_min + value_max) / 2
    for state in range(n_states):
        for action in range(n_actions):
            value = q[state, action]
            text_color = 'white' if value <= midpoint else 'black'
            ax.text(
                action,
                state,
                f'{value:.3f}',
                ha='center',
                va='center',
                color=text_color,
                fontsize=6,
            )

    fig.colorbar(image, ax=ax, label='Q-value')
    fig.tight_layout()
    fig.savefig(artifact_path('q_table.png'), dpi=160)
    plt.close(fig)


def visualize_saved_q_table():
    with open(artifact_path('q_table.pkl'), 'rb') as f:
        q = pickle.load(f)

    save_q_table_visualizations(q)
    print(
        'Saved visualizations: '
        f'{artifact_path("policy.png")}, '
        f'{artifact_path("q_table.png")}'
    )


def run(episodes, is_training=True, render=False):

    env = gym.make(
        'FrozenLake-v1',
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
        render_mode='human' if render else None,
    )

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open(artifact_path('q_table.pkl'), 'rb') as f:
            q = pickle.load(f)

        expected_shape = (env.observation_space.n, env.action_space.n)
        if q.shape != expected_shape:
            raise ValueError(
                f'Saved Q-table has shape {q.shape}, expected {expected_shape} '
                f'for the {MAP_NAME} map.'
            )

    learning_rate_a = 0.1      # alpha: how strongly new experience updates the Q-table
    discount_factor_g = 0.95   # gamma: how much the agent values future rewards
    epsilon = 1.0              # 1 = 100% random actions at the start of training
    min_epsilon = 0
    exploration_episodes = EPSILON_DECAY_EPISODES[MAP_NAME]
    epsilon_decay_rate = (epsilon - min_epsilon) / exploration_episodes
    rng = np.random.default_rng(TRAINING_SEED)
    env.action_space.seed(TRAINING_SEED)

    rewards_per_episode = np.zeros(episodes)
    progress_interval = max(1, episodes // 10)

    if is_training:
        before_wins, before_rate = evaluate_agent(None)
        print(f'Before training: {before_wins}/1000 wins ({before_rate:.1f}% success rate)')
        print(f'Training started: {episodes} episodes')
        print(
            f'Parameters: alpha={learning_rate_a}, gamma={discount_factor_g}, '
            f'epsilon={epsilon}->{min_epsilon} over {exploration_episodes} episodes'
        )

    for i in range(episodes):
        state = env.reset(seed=TRAINING_SEED if i == 0 else None)[0]
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = select_best_action(q, state, rng)

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                update_q_value(
                    q,
                    state,
                    action,
                    reward,
                    new_state,
                    terminated or truncated,
                    learning_rate_a,
                    discount_factor_g,
                )

            state = new_state

        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)

        if reward == 1:
            rewards_per_episode[i] = 1

        completed = i + 1
        if is_training and (completed % progress_interval == 0 or completed == episodes):
            wins = int(np.sum(rewards_per_episode[:completed]))
            recent_window = rewards_per_episode[max(0, completed - 1000):completed]
            recent_success_rate = np.mean(recent_window) * 100
            checkpoint_wins, checkpoint_rate = evaluate_agent(q, episodes=500)
            print(
                f'Progress: {completed}/{episodes} episodes | '
                f'total wins: {wins} | training rate: {recent_success_rate:.1f}% | '
                f'greedy policy test: {checkpoint_wins}/500 ({checkpoint_rate:.1f}%) | '
                f'epsilon: {epsilon:.3f}'
            )

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-99):(t+1)])

    if is_training:
        plt.figure()
        plt.plot(sum_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Wins in last 100 episodes')
        plt.title(f'FrozenLake {MAP_NAME} Q-learning training progress')
        plt.grid(True)
        plt.savefig(artifact_path('training_progress.png'))
        plt.close()

        with open(artifact_path('q_table.pkl'), 'wb') as f:
            pickle.dump(q, f)

        save_q_table_visualizations(q)

        after_wins, after_rate = evaluate_agent(q)
        metrics = {
            'map_name': MAP_NAME,
            'is_slippery': IS_SLIPPERY,
            'episodes': episodes,
            'alpha': learning_rate_a,
            'gamma': discount_factor_g,
            'initial_epsilon': 1.0,
            'minimum_epsilon': min_epsilon,
            'epsilon_decay_episodes': exploration_episodes,
            'training_seed': TRAINING_SEED,
            'evaluation_seed': EVALUATION_SEED,
            'evaluation_episodes': 1000,
            'before_training_wins': before_wins,
            'before_training_success_rate': before_rate,
            'after_training_wins': after_wins,
            'after_training_success_rate': after_rate,
        }
        with open(artifact_path('metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        print(f'After training: {after_wins}/1000 wins ({after_rate:.1f}% success rate)')
        print(
            'Saved: '
            f'{artifact_path("q_table.pkl")}, '
            f'{artifact_path("training_progress.png")}, '
            f'{artifact_path("policy.png")}, '
            f'{artifact_path("q_table.png")}, '
            f'{artifact_path("metrics.json")}'
        )


def main():
    global MAP_NAME, IS_SLIPPERY

    parser = argparse.ArgumentParser(
        description='Train, evaluate, or demonstrate a FrozenLake Q-learning agent.'
    )
    parser.add_argument('mode', choices=('train', 'demo', 'visualize'))
    parser.add_argument('--map-name', choices=tuple(MAP_LAYOUTS), default='8x8')
    ice_group = parser.add_mutually_exclusive_group()
    ice_group.add_argument('--slippery', dest='slippery', action='store_true')
    ice_group.add_argument('--no-slippery', dest='slippery', action='store_false')
    parser.set_defaults(slippery=True)
    parser.add_argument('--episodes', type=int, default=40000)
    parser.add_argument('--demo-episodes', type=int, default=3)
    args = parser.parse_args()

    MAP_NAME = args.map_name
    IS_SLIPPERY = args.slippery

    if args.mode == 'train':
        run(args.episodes)
    elif args.mode == 'demo':
        run(args.demo_episodes, is_training=False, render=True)
    else:
        visualize_saved_q_table()


if __name__ == '__main__':
    main()
