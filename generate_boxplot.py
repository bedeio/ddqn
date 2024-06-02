import torch
import os
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from dqn import QNetwork
from rl import train_dagger, test_policy_full
from dt import DTPolicy, load_dt_policy, save_dt_policy, save_dt_policy_viz

def _w(model_type, max_depth, is_reweight):
    return {
        "model_type": model_type,
        "max_depth": max_depth,
        "is_reweight": is_reweight
    }

def get_filename(env_name, w):
    file_name_suffix = f'{w["model_type"]}_depth{w["max_depth"]}_reweight{w["is_reweight"]}'
    return f"dt_policy_{env_name}_{file_name_suffix}.pk"

def teacher_paths(args):
    # Initialize a dictionary to hold the paths for each environment
    env_paths = {env: [] for env in args.env_names}

    for env in args.env_names:
        mapped_env = "CartPole-v1" if env == "WindyCartPole-v1" else env
        path = f'{args.model_directory}/checkpoint_{mapped_env}.pth'
        env_paths[env].append(path)

    print(env_paths)
    return env_paths


def student_paths(args):
    env_paths = {env: [] for env in args.env_names}

    for env in args.env_names:
        mapped_env = "CartPole-v1" if env == "WindyCartPole-v1" else env
        for file in sorted(os.listdir(f'{args.model_directory}/trees')):
            if file.endswith('.pk') and file.startswith('dt_policy') and mapped_env in file:
                full_path = f'{args.model_directory}/trees/{file}'
                env_paths[env].append(full_path)

def load_ddqn(env_name, dir):
    q_network = QNetwork(state_size=8, action_size=4, seed=42)
    q_network.load_state_dict(
        torch.load(f'{dir}/checkpoint_{env_name}.pth', map_location=torch.device('cpu')))

    return q_network


def compare_all(config, winners, save_suffix, n=1_000):
    env = gym.make(
        'LunarLander-v2',
        enable_wind=True,
        wind_power=config["wind_power"],
        turbulence_power=config["turbulence_power"]
    )

    print("Loading models...")
    policies = {
        'ddqn': load_ddqn(config["env_name"], config["teacher_dir"])
    }

    for winner in winners:
        model_type = winner["model_type"]
        max_depth = winner["max_depth"]
        is_reweight = winner["is_reweight"]
        fname = get_filename(config["env_name"], winner)
        
        key = f'{model_type}\nDepth: {max_depth}\nReweight: {is_reweight}'
        policies[key] = load_dt_policy(config["student_dir"], fname)

    rewards = {}
    for model in policies:
        no_newln = model.replace("\n", "_")
        rewards[model] = test_policy_full(
            env, policies[model], n_test_rollouts=n)
        print(f"{no_newln} - Avg. Reward: {np.mean(rewards[model])}")
        print("---")

    df = pd.DataFrame.from_dict(rewards)
    print(df.head())

    plt.figure(figsize=(9, 5))
    sns.violinplot(data=df, palette='tab10')
    plt.savefig(f'plots/violin_{save_suffix}.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        palette='tab10',
        showmeans=True,
        meanprops={"marker": "o",
                   "markerfacecolor": "white",
                   "markeredgecolor": "black",
                   "markersize": "8"})
    plt.savefig(f'plots/box_{save_suffix}.png')
    plt.show()


# taken from get_top_n.py for the robust lunar lander results
winners_turb_no_wind = [
    _w("linear_tree_logistic", 6, True),
    _w("linear_tree_logistic", 1, False),
    _w("linear_tree_ridge", 6, True)
]

winners_turb_max_wind = [
    _w("linear_tree_logistic", 6, True),
    _w("linear_tree_logistic", 1, False),
    _w("linear_tree_ridge", 4, True)
]

base_config = {
    "env_name": "LunarLander-v2",
    "teacher_dir": "models",
    "student_dir": "models/trees"
}

config_turb_no_wind = base_config.copy()
config_turb_no_wind.update({
    "turbulence_power": 2.0,
    "wind_power": 0.0
})

config_turb_max_wind = base_config.copy()
config_turb_max_wind.update({
    "turbulence_power": 2.0,
    "wind_power": 20.0
})

if __name__ == '__main__':
    compare_all(config_turb_no_wind, winners_turb_no_wind, "turb_no_wind")
    compare_all(config_turb_max_wind, winners_turb_max_wind, "turb_max_wind")