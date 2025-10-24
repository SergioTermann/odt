import gym_dogfight
import argparse
import numpy as np
import pickle
import os
import time
from datetime import datetime
import random
import torch
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def argsparser():
    parser = argparse.ArgumentParser("Strategy Data Collection")
    parser.add_argument('--host', default='192.168.43.180', metavar='str', help='specifies Harfang host id')
    # parser.add_argument("--host", type=str, default='172.27.240.1')
    parser.add_argument('--port', default='57805', metavar='str', help='specifies Harfang port id')
    parser.add_argument('--env_id', help='environment ID', default='data_collection-v0')
    parser.add_argument('--episodes', type=int, help='Number of episodes to collect', default=10)
    parser.add_argument('--policy_path', type=str, help='Path to policy model', default=None)
    parser.add_argument('--output_dir', type=str, help='Directory to save results', default='collected_data')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    return parser.parse_args()


class EpisodeCollector:
    def __init__(self, env, policy=None):
        self.env = env
        self.policy = policy
        self.trajectory_length = env.trajectory_length

    def calculate_rtg(self, rewards):
        rtg = np.zeros_like(rewards, dtype=float)
        rtg[-1] = rewards[-1]

        # Compute RTG in reverse order
        for t in reversed(range(len(rewards) - 1)):
            rtg[t] = rewards[t] + rtg[t + 1]

        return rtg

    def collect_episode(self):
        current_state = self.env.reset()

        observations = []
        next_observations = []
        actions = []
        rewards = []
        terminals = []
        rtgs = []  # New list to store Return-to-Go values

        for step in range(self.trajectory_length):
            action_base = None
            next_state, reward, done, action = self.env.steps(action_base)

            # Store the transition
            observations.append(current_state.copy())
            actions.append(action.copy())
            next_observations.append(next_state.copy())
            rewards.append(reward)
            terminals.append(done)

            if done:
                # If episode is done, we still want to pad the arrays to maintain consistent size
                remaining_steps = self.trajectory_length - step - 1
                if remaining_steps > 0:
                    # Pad with zeros for observations and actions, True for terminals
                    observations.extend([np.zeros_like(current_state)] * remaining_steps)
                    actions.extend([np.zeros_like(action)] * remaining_steps)
                    next_observations.extend([np.zeros_like(next_state)] * remaining_steps)
                    rewards.extend([0.0] * remaining_steps)
                    terminals.extend([True] * remaining_steps)
                break

            current_state = next_state

        # Calculate Return-to-Go and reshape to (2048, 1)
        rtgs = self.calculate_rtg(np.array(rewards)).reshape(-1, 1)

        # Create episode data dictionary
        episode_data = {
            'observations': np.array(observations),
            'next_observations': np.array(next_observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'terminals': np.array(terminals),
            'rtgs': rtgs,  # RTGs now in (2048, 1) shape
        }

        return episode_data


def load_policy(path, env):

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(state_dim, action_dim, max_action)
    policy.load(path)

    return policy


def main(args):
    # Create environment
    env = gym_dogfight.make(args.env_id, host=args.host, port=args.port, rendering=True)

    # Load policy if specified
    policy = None
    if args.policy_path is not None:
        policy = load_policy(args.policy_path, env)

    # Create collector
    collector = EpisodeCollector(env, policy)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Collect episodes
    all_episodes = {}
    output_filename = os.path.join(args.output_dir, f"episodes_{timestamp}.pkl")
    print(f"Collecting {args.episodes} episodes...")

    for episode_idx in tqdm(range(args.episodes)):
        # Collect episode
        episode_data = collector.collect_episode()

        # Add to collection with the format shown in the screenshot (0000, 0001, etc.)
        all_episodes[f"{episode_idx:04d}"] = episode_data

        # Print progress every 100 episodes
        if (episode_idx + 1) % 100 == 0:
            print(f"Collected {episode_idx + 1}/{args.episodes} episodes")

        if (episode_idx + 1) % 20 == 0:
            output_filename = os.path.join(args.output_dir, f"episodes_{timestamp}.pkl")
            with open(output_filename, 'wb') as f:
                pickle.dump(all_episodes, f)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Save final episodes if not already saved
    with open(output_filename, 'wb') as f:
        pickle.dump(all_episodes, f)
    print(f"All episodes saved to {output_filename}")


if __name__ == '__main__':
    args = argsparser()
    main(args)
