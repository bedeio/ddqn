# -----------------------------
# Import the Necessary Packages
# -----------------------------
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io
import numpy as np
from collections import deque, namedtuple
import scipy.stats as stats

# For visualization
from gymnasium.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display
import glob


# --------------------------
# Initialize the environment
# --------------------------
env = gym.make('LunarLander-v2')
#env.seed(0)
#print('State shape: ', env.observation_space.shape)
#print('Number of actions: ', env.action_space.n)

# ----------------------------------
# Define Neural Network Architecture
# ----------------------------------
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

# --------------------------
# Define some hyperparameter
# --------------------------
DDQN = True             # DDQN = False for DQN and True for Double DQN
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
#BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
#TAU = 0.1
# LR = 5e-4               # learning rate
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
#UPDATE_EVERY = 8        # how often to update the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print('device:',device)

# ------------
# Define Agent
# ------------
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = 42 #random.seed(seed)

        # Q-Network
        # from ddt import DDT
        # self.qnetwork_local = DDT(alpha=1, input_dim=8, output_dim=4, leaves=32, is_value=True, weights=None, comparators=None)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        # self.qnetwork_target = DDT(alpha=1, input_dim=8, output_dim=4, leaves=32, is_value=True, weights=None, comparators=None)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss

        if DDQN == True: # Double DQN
            ### Extract next action indices using the local network
            next_actions = self.qnetwork_local(next_states).argmax(1).unsqueeze(1)
            ### Evaluate next state's Q-values using the target network
            q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions).detach()
        else:            # DQN
            ### Extract next maximum estimated value from target network
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        ### Calculate target value from Bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        ### Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)

        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    # ----------------------------------------------------------
    # Validation Process do Agent
    # Retorna a média do score e o intervalo de confiança de 95%
    # ----------________________________________________________
    def validation(self,num_evaluations):
        rewards = []
        self.qnetwork_local.eval()
        with torch.no_grad():
            for _ in range(num_evaluations):
                state = np.array(env.reset()[0])
                done = False
                reward_acum = 0
                while not done:
                    action = agent.act(state)
                    state, reward, done, *extra_vars = env.step(action)
                    reward_acum = reward_acum + reward
                    if done or reward_acum <= -250:
                        rewards.append(reward_acum)
                        break

        # Calculate rewards mean
        average = np.mean(rewards)

        # Calculate 95% confidence interval
        confidence_interval = stats.t.interval(0.95, len(rewards)-1, loc=average, scale=stats.sem(rewards))

        self.qnetwork_local.train()
        env.close()
        return average, confidence_interval

# --------------------
# Define Replay Buffer
# --------------------
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device, non_blocking=True)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device, non_blocking=True)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device, non_blocking=True)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device, non_blocking=True)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device, non_blocking=True)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# ----------------
# Training Process
# ----------------
def dqn(n_episodes=1500, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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
            score_avg, score_interval = agent.validation(num_evaluations=30)
            print(f'\tValidation (Min,Avg,Max): \t{score_interval[0]:.2f},\t{score_avg:.2f},\t{score_interval[1]:.2f}')
            episodes_list.append(i_episode)
            score_avg_list.append(score_avg)
            score_interval_list.append(score_interval)

        if np.mean(scores_window) >= 250.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,np.mean(scores_window)))
            score_avg, score_interval = agent.validation(num_evaluations=30)
            print(f'\tValidation Min:Avg:Max \t{score_interval[0]:.2f}:\t{score_avg:.2f}:\t{score_interval[1]:.2f}')
            episodes_list.append(i_episode)
            score_avg_list.append(score_avg)
            score_interval_list.append(score_interval)
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    return scores, episodes_list, score_avg_list, score_interval_list

print(device)
agent = Agent(state_size=8, action_size=4, seed=42)
scores, episodes_list, score_avg_list, score_interval_list = dqn()


# --------------------------
# Plot the learning progress
# --------------------------
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

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
plt.show()

# ---------------------
# Animate it with Video
# ---------------------

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode="human")
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    state = np.array(env.reset()[0])
    done = False
    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, *extra_vars = env.step(action)
    env.close()

# agent = Agent(state_size=8, action_size=4, seed=42)
show_video_of_model(agent, 'LunarLander-v2')

