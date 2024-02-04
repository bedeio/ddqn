import gymnasium as gym
import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from agent import Agent, Config
from gymnasium.envs.registration import register
from env.windy_cartpole_env import WindyCartPole

# For visualization
from gymnasium.wrappers import transform_observation as to
from gymnasium.wrappers.monitoring import video_recorder
from IPython.display import HTML

def save_model(agent, env_name, models_dir='models'):
    print(f"Saving model for {env_name}")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    torch.save(agent.qnetwork_local.state_dict(), f'{models_dir}/checkpoint_{env_name}.pth')

def train_dqn(agent, env_name, max_score, n_episodes=7500, max_t=2500, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    scores = []
    episodes_list = []
    score_avg_list = []
    score_interval_list = []
    scores_window = deque(maxlen=100) 
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state = np.array(env.reset()[0])
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, *extra_vars = env.step(action)
            agent.step(state, action, reward, next_state, terminated or truncated)
            state = next_state
            score += reward
            if terminated or truncated:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        score_avg = 0

        print('\r', 'DDQN Training   Episode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\r', 'DDQN Training   Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)), end="")
            score_avg, score_interval = agent.validation(num_evaluations=30)
            print(
                f'\tValidation (Min,Avg,Max): \t{score_interval[0]:.2f},\t{score_avg:.2f},\t{score_interval[1]:.2f}')
            episodes_list.append(i_episode)
            score_avg_list.append(score_avg)
            score_interval_list.append(score_interval)

        if score_avg >= max_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))

            score_avg, score_interval = agent.validation(num_evaluations=30)
            print(
                f'\tValidation Min:Avg:Max \t{score_interval[0]:.2f}:\t{score_avg:.2f}:\t{score_interval[1]:.2f}')
            episodes_list.append(i_episode)
            score_avg_list.append(score_avg)
            score_interval_list.append(score_interval)
            break

    save_model(agent, env_name)
    return scores, episodes_list, score_avg_list, score_interval_list


def plot_scores(scores, filename='plots/scores_plot.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    plt.savefig(filename)
    plt.show()


def plot_validation_progress(episodes_list, score_avg_list, filename='plots/ddqn_validation_plot.png'):
    plt.plot(episodes_list, score_avg_list, label='Average Score')
    interval_min = [i[0] for i in score_interval_list]
    interval_max = [i[1] for i in score_interval_list]
    plt.fill_between(episodes_list, interval_min, interval_max,
                     alpha=0.3, label='Confidence Interval 95%')

    plt.title('Validation')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend()

    plt.savefig(filename)
    plt.show()


def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode="human")
    # agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    state = np.array(env.reset()[0])
    done = False
    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, *extra_vars = env.step(action)
    env.close()

def get_env_params(env):
    state_size = env.observation_space.shape[0] if isinstance(env.observation_space, gym.spaces.Box) else env.observation_space.n
    action_size = env.action_space.n

    return state_size, action_size

def decode_state(encoded_state):
    destination = encoded_state % 4
    state = encoded_state // 4
    
    passenger_location = state % 5
    state = state // 5
    taxi_col = state % 5
    taxi_row = state // 5
    
    # One-hot encoding
    passenger_location_encoded = np.zeros(5)
    passenger_location_encoded[passenger_location] = 1

    destination_encoded = np.zeros(4)
    destination_encoded[destination] = 1

    return np.concatenate(([taxi_row, taxi_col], passenger_location_encoded, destination_encoded))

def select_config(env_name, configs):
    for key in configs:
        if env_name.startswith(key):
            return configs[key]
        
    raise ValueError(f"No configuration found for environment: {env_name}")


def build_env(env_name, **kwargs):
    if env_name == "WindyCartPole-v1":
        register(
            id='WindyCartPole-v1',
            entry_point='env.windy_cartpole_env:WindyCartPole',
            max_episode_steps=2500,
            reward_threshold=500,
        )
        env = gym.make(env_name, **kwargs)
    else:
        env = gym.make(env_name, **kwargs)
    
    state_size, action_size = get_env_params(env)
    
    if env_name.startswith("Taxi"):
        state_size = 11
        env = to.TransformObservation(env, lambda obs: decode_state(obs))

    return env, state_size, action_size

agent_configs = {
    "WindyCartPole": Config(DDQN=True, BUFFER_SIZE=int(7e5), BATCH_SIZE=512, GAMMA=0.99, TAU=1e-2, LR=1e-4, UPDATE_EVERY=4, LOSS=F.mse_loss),
    "CartPole": Config(DDQN=True, BUFFER_SIZE=int(7e5), BATCH_SIZE=512, GAMMA=0.99, TAU=1e-2, LR=1e-4, UPDATE_EVERY=4, LOSS=F.mse_loss),
    "LunarLander": Config(DDQN=True, BUFFER_SIZE=int(7e5), BATCH_SIZE=512, GAMMA=0.99, TAU=1e-2, LR=1e-3, UPDATE_EVERY=4, LOSS=F.smooth_l1_loss),
    "Taxi": Config(DDQN=True, BUFFER_SIZE=int(7e5), BATCH_SIZE=512, GAMMA=0.98, TAU=1e-2, LR=4e-3, UPDATE_EVERY=4, LOSS=F.mse_loss)
}

solved_scores = {
    # https://github.com/openai/gym/wiki/Leaderboard
    # https://gymnasium.farama.org/environments/box2d/lunar_lander/#rewards
    "WindyCartPole": 400,
    "CartPole": 400,
    "LunarLander": 250, 
    "Taxi": 8.5
}

# Options: 'LunarLander-v2', 'Taxi-v3', 'CartPole-v1'
# env_name = 'LunarLander-v2'  
# config = select_config(env_name, agent_configs)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':    
    default_env_names = ['LunarLander-v2', 'Taxi-v3', 'CartPole-v1']
    env_names = sys.argv[1:] if len(sys.argv) > 1 else default_env_names

    for env_name in env_names:
        env, state_size, action_size = build_env(env_name)
        print(f"Environment: {env_name}, State Size: {state_size}, Action Size: {action_size}")

        conf = select_config(env_name, agent_configs)
        agent = Agent(env, state_size=state_size, action_size=action_size, config=conf)

        max_score = select_config(env_name, solved_scores)
        scores, episodes_list, score_avg_list, score_interval_list = train_dqn(agent, env_name, max_score)

        # plot_validation_progress(episodes_list, score_avg_list)
        # plot_scores(scores)

        # show_video_of_model(agent, env_name)`
