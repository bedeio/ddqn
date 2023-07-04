from rl import test_policy_full

import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from dqn import QNetwork
from rl import train_dagger, test_policy_full
from dt import DTPolicy, load_dt_policy, save_dt_policy, save_dt_policy_viz


def load_ddqn():
    q_network = QNetwork(state_size=8, action_size=4, seed=42)
    q_network.load_state_dict(
        torch.load('checkpoint.pth', map_location=torch.device('cpu')))

    return q_network


def compare_all():
    env = gym.make(
        'LunarLander-v2',
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=1.5
    )
    n = 1000

    print("Loading models...")
    policies = {
        'ddqn': load_ddqn(),
        'simple_dt': load_dt_policy('models', 'simple_dt_policy.pk'),
        'linear_dt': load_dt_policy('models', 'linear_dt_policy.pk'),
        'logistic_dt': load_dt_policy('models', 'logistic_dt_policy.pk'),
    }

    rewards = {}
    for model in policies:
        print("Testing model:", model)
        rewards[model] = test_policy_full(
            env, policies[model], n_test_rollouts=n)
        print(f"{model} - Avg. Reward: {np.mean(rewards[model])}")

    df = pd.DataFrame.from_dict(rewards)
    print(df.head())
    plt.figure(figsize=(9, 5))
    sns.violinplot(data=df, palette='tab10')
    plt.savefig('comp.png')
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
    plt.savefig('box_comp.png')
    plt.show()


if __name__ == '__main__':
    compare_all()
