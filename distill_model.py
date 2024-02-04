import argparse
import csv
import itertools
from dataclasses import dataclass
import torch
import gymnasium as gym
import os

from dqn import QNetwork
from rl import train_dagger, test_policy
from dt import DTPolicy, load_dt_policy, save_dt_policy, save_dt_policy_viz 
from train import build_env

@dataclass
class Config:
    log_fname: str
    n_batch_rollouts: int
    max_samples: int
    max_iters: int
    train_frac: float
    is_reweight: bool
    n_test_rollouts: int
    save_dirname: str
    save_fname: str
    save_viz_fname: str
    is_train: bool
    env_name: str
    tree_type: str
    max_depths: list
    depth: int

def load_teacher(env_name, state_size, action_size):
    qnet = QNetwork(state_size=state_size, action_size=action_size, seed=42)
    qnet.load_state_dict(
        torch.load(f'models/checkpoint_{env_name}.pth', map_location=torch.device('cpu')))
    
    return qnet

def learn_dt(config: Config):
    print(f"Learning {config.tree_type} with depth {config.depth} and is_reweight {config.is_reweight}")
    # Data structures
    env, state_size, action_size = build_env(config.env_name)
    teacher = load_teacher(config.env_name, state_size, action_size)

    student = DTPolicy(config.depth, config.tree_type)

    # Train student
    if config.is_train:
        student = train_dagger(env, teacher, student, config.max_iters, config.n_batch_rollouts, 
                               config.max_samples, config.train_frac, config.is_reweight, config.n_test_rollouts)
        save_dt_policy(student, config.save_dirname, config.save_fname)
        # save_dt_policy_viz(student, config.save_dirname, config.save_viz_fname)
    else:
        student = load_dt_policy(config.save_dirname, config.save_fname)

    # Test student
    rew = test_policy(env, student, config.n_test_rollouts)
    print('Final reward: {}'.format(rew))
    
    return rew 

def parse_args():
    parser = argparse.ArgumentParser(description="Train and test a decision tree policy.")
    parser.add_argument("--log_fname", type=str, default='logs/dt.log')
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n_batch_rollouts", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=200_000)
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--is_reweight", type=bool, default=True)
    parser.add_argument("--n_test_rollouts", type=int, default=30)
    parser.add_argument("--save_dirname", type=str, default='models/trees')
    parser.add_argument("--save_fname", type=str, default='linear_dt_policy.pk')
    parser.add_argument("--save_viz_fname", type=str, default='linear_dt_policy.dot')
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--tree_type", type=str)
    parser.add_argument("--max_depths", type=int, nargs='*', default=[1, 2, 4, 6, 8],
                        help="List of maximum depths to try for the decision tree. Defaults to [2, 4, 6, 8, 10].")

    return parser.parse_args()

def main():
    args = parse_args()
    config = Config(**vars(args))

    # tree_types = ['decision_tree', 'linear_tree_ridge', 'linear_tree_logistic']

    if not config.tree_type:
        generate_data(config)
    else:
        learn_dt(config)

def generate_data(config):
    max_depths = config.max_depths #[1,2, 4, 6, 8]
    is_reweight_options = [True, False]
    tree_types = ['decision_tree', 'linear_tree_ridge', 'linear_tree_logistic']

    all_combinations = list(itertools.product(max_depths, is_reweight_options, tree_types))
    total_combinations = len(all_combinations)
    results = []

    for idx, (depth, is_reweight, tree_type) in enumerate(all_combinations, start=1):
        file_name_suffix = f"{tree_type}_depth{depth}_reweight{is_reweight}"
        save_fname = f"dt_policy_{config.env_name}_{file_name_suffix}.pk"
        save_viz_fname = f"dt_policy_{config.env_name}_{file_name_suffix}.dot"

        modified_config_dict = vars(config).copy()
        modified_config_dict.update({
            'depth': depth,
            'is_reweight': is_reweight,
            'tree_type': tree_type,
            'save_fname': save_fname,
            'save_viz_fname': save_viz_fname,
        })

        config = Config(**modified_config_dict)
        fpath = f"{config.save_dirname}/{save_fname}"
        if os.path.exists(fpath):
            print(f"Skipping {config.tree_type} with depth={config.depth} and reweight={config.is_reweight} ")
            continue

        rew = learn_dt(config)

        # Append results
        results.append({
            'depth': depth,
            'is_reweight': is_reweight,
            'tree_type': tree_type,
            'reward': rew
        })

        # Calculate and print the percentage of completion
        percentage_completed = (idx / total_combinations) * 100
        print(f"\rProgress: {percentage_completed:.2f}%", end="")

    # Move to the next line after completing all iterations
    print()

    # Export results to a CSV file
    with open(f'csv/dt_policy_results_{config.env_name}.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['depth', 'is_reweight', 'tree_type', 'reward'])
        writer.writeheader()
        writer.writerows(results)

if __name__ == '__main__':
    main()
