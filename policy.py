import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from ddt import DDT
from sdt import SDT

class ActorCriticNet(nn.Module):
    def __init__(self):
        super(ActorCriticNet, self).__init__()
        hidden_size = 128
        self.affine = nn.Linear(8, hidden_size)
        
        # self.action_layer = DDT(alpha=1, input_dim=hidden_size, output_dim=4, leaves=64, is_value=True, weights=None, comparators=None)
        self.action_layer = nn.Linear(hidden_size, 4)
        self.value_layer = nn.Linear(hidden_size, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state.unsqueeze(0)), dim=-1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
    
    def calculateLoss(self, gamma=0.99):
        
        # rewards = []
        # dis_reward = 0
        # for reward in self.rewards[::-1]:
        #     dis_reward = reward + gamma * dis_reward
        #     rewards.insert(0, dis_reward)
                
        # rewards = torch.tensor(rewards, requires_grad=True).float()
        # rewards = (rewards - rewards.mean()) / (rewards.std())
        
        # logprobs = torch.tensor(self.logprobs, requires_grad=False)
        # state_vals = torch.tensor(self.state_values, requires_grad=False)
        # # state_vals = (state_vals - state_vals.mean()) / (state_vals.std())

        # advantages = rewards - state_vals
        # # advantages = (advantages - advantages.mean()) / (advantages.std())
        # action_loss = -logprobs * advantages
        # value_loss = F.huber_loss(state_vals, rewards)
        
        # return value_loss.sum() + action_loss.sum()
    
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

def train():
    # Defaults parameters:
    #    gamma = 0.99
    #    lr = 0.02
    #    betas = (0.9, 0.999)
    #    random_seed = 543

    render = False
    gamma = 0.99
    lr = 0.01
    random_seed = 42
    
    torch.manual_seed(random_seed)
    
    env = gym.make('LunarLander-v2')
    # env.seed(random_seed)
    
    policy = ActorCriticNet()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    running_reward = 0
    for i_episode in range(0, 10000):
        state = np.array(env.reset()[0])
        for t in range(10000):
            action = policy(state)
            state, reward, done, *ext = env.step(action)
            policy.rewards.append(reward)
            running_reward += reward
            if render and i_episode > 1000:
                env.render()
            if done:
                break
                    
        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()        
        policy.clearMemory()
        
        # saving the model if episodes > 999 OR avg reward > 200 
        #if i_episode > 999:
        #    torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
        
        if running_reward > 4000:
            # torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("########## Solved! ##########")
            # test(name='LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            break
        
        if i_episode % 20 == 0:
            running_reward = running_reward/20
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0
            
if __name__ == '__main__':
    train()