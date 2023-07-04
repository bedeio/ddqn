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


policies = {
    'ddqn': load_ddqn(),
    'simple_dt': load_dt_policy('models', 'simple_dt_policy.pk'),
    'linear_dt': load_dt_policy('models', 'linear_dt_policy.pk'),
    'logistic_dt': load_dt_policy('models', 'logistic_dt_policy.pk'),
}

n = 10


def compare_all():
    wind_powers = [0.0, 5.0, 10.0, 15.0, 20.0]
    turbulence = 1.5
    rewards = []

    for wp in wind_powers:
        env = gym.make(
            'LunarLander-v2',
            enable_wind=True,
            wind_power=wp,
            turbulence_power=turbulence
        )
        print(f"\nTesting for wind power: {wp}")

        for model in policies:
            print("Testing model:", model)
            reward = np.mean(test_policy_full(
                env, policies[model], n_test_rollouts=n))
            print(f"{model} - Avg. Reward: {reward}")
            rewards.append([wp, model, reward])

    # Convert rewards to DataFrame
    df_mean_rewards = pd.DataFrame(
        rewards, columns=['Wind Power', 'Model', 'Avg. Reward'])

    # Pivot the DataFrame so that 'Model' values become rows and 'Wind Power' values become columns
    df_mean_rewards_pivot = df_mean_rewards.pivot(
        index='Model', columns='Wind Power', values='Avg. Reward')

    # Add "wind_power=" to the column names
    df_mean_rewards_pivot.columns = [
        'wind_power=' + str(col) for col in df_mean_rewards_pivot.columns]

    # Round the rewards to 2 decimal places
    df_mean_rewards_pivot = df_mean_rewards_pivot.round(2)

    # Create a new figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Hide axes
    ax.axis('tight')
    ax.axis('off')

    # Create the table and scale it to the figure
    the_table = ax.table(cellText=df_mean_rewards_pivot.values,
                         colLabels=df_mean_rewards_pivot.columns,
                         rowLabels=df_mean_rewards_pivot.index,
                         cellLoc='center', loc='center')

    # Save the figure before showing it
    name = 'table.png' if turbulence == 0 else 'table_with_turbulence.png'
    plt.savefig(name, bbox_inches='tight')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    compare_all()
