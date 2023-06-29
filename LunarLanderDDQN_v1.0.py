# -----------------------------
# Import the Necessary Packages
# -----------------------------
import gymnasium as gym
import random
import torch

import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io
import numpy as np
from collections import deque
import scipy.stats as stats

# For visualization
from gymnasium.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display
import glob

from agent import Agent, ReplayBuffer
from dqn import QNetwork


# --------------------------
# Initialize the environment
# --------------------------
#env.seed(0)
#print('State shape: ', env.observation_space.shape)
#print('Number of actions: ', env.action_space.n)

# --------------------------
# Define some hyperparameter
# --------------------------
DDQN = True             # DDQN = False for DQN and True for Double DQN
BUFFER_SIZE = int(2e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
#BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
#TAU = 0.1
# LR = 5e-4               # learning rate
LR = 1e-3               # learning rate
UPDATE_EVERY = 4        # how often to update the network
#UPDATE_EVERY = 8        # how often to update the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print('device:',device)

# ----------------
# Training Process
# ----------------
def dqn(agent, n_episodes=1500, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    episodes_list = []  # list containing episodes of evaluations
    score_avg_list = []  # list containing score average of evaluations
    score_interval_list = []  # list containing score interval of evaluations
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = np.array(env.reset()[0])
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, *extra_vars = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        tipo = "DDQN" if DDQN else "DQN"
        print('\r',tipo,'Training   Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\r',tipo,'Training   Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            score_avg, score_interval = agent.validation(num_evaluations=5)
            print(f'\tValidation (Min,Avg,Max): \t{score_interval[0]:.2f},\t{score_avg:.2f},\t{score_interval[1]:.2f}')
            episodes_list.append(i_episode)
            score_avg_list.append(score_avg)
            score_interval_list.append(score_interval)

        if np.mean(scores_window) >= 250.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,np.mean(scores_window)))
            # score_avg, score_interval = agent.validation(num_evaluations=30)
            score_avg, score_interval = agent.validation(num_evaluations=30)
            print(f'\tValidation Min:Avg:Max \t{score_interval[0]:.2f}:\t{score_avg:.2f}:\t{score_interval[1]:.2f}')
            episodes_list.append(i_episode)
            score_avg_list.append(score_avg)
            score_interval_list.append(score_interval)
            torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoint.pth')
            break

    return scores, episodes_list, score_avg_list, score_interval_list



def plot_scores(scores, filename='graphs/scores_plot.png'):
    # --------------------------
    # Plot the learning progress
    # --------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    plt.savefig(filename)
    plt.show()

def plot_validation_progress(episodes_list, score_avg_list, filename='graphs/validation_plot.png'):
    # ----------------------------
    # Plot the validation progress
    # ----------------------------
    # Criação do gráfico
    plt.plot(episodes_list, score_avg_list, label='Average Score')
    interval_min = [i[0] for i in score_interval_list]
    interval_max = [i[1] for i in score_interval_list]
    plt.fill_between(episodes_list, interval_min, interval_max, alpha=0.3, label='Confidence Interval 95%')
    # Títulos e legendas
    plt.title('Validation')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend()

    plt.savefig(filename)
    plt.show()



def show_video_of_model(agent, env_name):
    # ---------------------
    # Animate it with Video
    # ---------------------
    env = gym.make(env_name, render_mode="human")
    #agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    state = np.array(env.reset()[0])
    done = False
    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, *extra_vars = env.step(action)
    env.close()


if __name__ == '__main__':
    # env = gym.make('LunarLander-v2', render_mode="human")
    env = gym.make('LunarLander-v2')
    # print('Running on device:', device)
    agent = Agent(env, state_size=8, action_size=4, seed=42, GAMMA=GAMMA, TAU=TAU, LR=LR, UPDATE_EVERY=UPDATE_EVERY, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE, DDQN=DDQN)
    scores, episodes_list, score_avg_list, score_interval_list = dqn(agent)
    plot_validation_progress(episodes_list, score_avg_list)
    plot_scores(scores)
    # agent = Agent(state_size=8, action_size=4, seed=42)
    show_video_of_model(agent, 'LunarLander-v2')

