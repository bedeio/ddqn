import scipy.stats as stats

import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from dqn import QNetwork
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import Callable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    GAMMA: float
    TAU: float
    LR: float
    UPDATE_EVERY: int
    BATCH_SIZE: int
    BUFFER_SIZE: int
    DDQN: bool
    LOSS: Callable


class Agent():
    def __init__(self, env, state_size, action_size, config: Config, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = 42  # random.seed(seed)
        self.env = env
        self.config = config
        self.steps = 0
        self.t_step = 0

        # Replay memory
        self.memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, seed)

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.LR)
        

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.config.GAMMA)

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

        # Compute and minimize the loss

        if self.config.DDQN == True:  # Double DQN
            # Extract next action indices using the local network
            next_actions = self.qnetwork_local(
                next_states).argmax(1).unsqueeze(1)
            # Evaluate next state's Q-values using the target network
            q_targets_next = self.qnetwork_target(
                next_states).gather(1, next_actions).detach()
        else:            # DQN
            # Extract next maximum estimated value from target network
            q_targets_next = self.qnetwork_target(
                next_states).detach().max(1)[0].unsqueeze(1)

        # Calculate target value from Bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        # Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = self.config.LOSS(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config.TAU)
        # self.hard_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def validation(self, num_evaluations):
        rewards = []
        self.qnetwork_local.eval()
        with torch.no_grad():
            for _ in range(num_evaluations):
                state = np.array(self.env.reset()[0])
                done = False
                reward_acum = 0
                while not done:
                    action = self.act(state)
                    state, reward, terminated, truncated, info = self.env.step(
                        action)
                    reward_acum = reward_acum + reward
                    if terminated or truncated or reward_acum <= -500:
                        rewards.append(reward_acum)
                        break

        average = np.mean(rewards)
        confidence_interval = stats.t.interval(
            0.95, len(rewards)-1, loc=average, scale=stats.sem(rewards))

        self.qnetwork_local.train()
        self.env.close()
        return average, confidence_interval


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(device, non_blocking=True)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).long().to(device, non_blocking=True)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(device, non_blocking=True)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device, non_blocking=True)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device, non_blocking=True)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
