import argparse
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from agent import Agent, Config
from train import select_config, build_env
from rl import get_rollout
from dqn import QNetwork
from distill_model import load_teacher, load_dt_policy

def test_model(policy, env, n_runs=50):
    scores = []
    for _ in range(n_runs):
        state = np.array(env.reset()[0])
        score = 0
        done = False
        while not done:
            action = int(policy.predict(np.array([state]))[0])
            state, reward, done, terminated, *extra_vars = env.step(action)
            score += reward
            if done or terminated:
                break
        scores.append(score)

    return scores

agent_configs = {
    "CartPole": Config(DDQN=True, BUFFER_SIZE=int(5e5), BATCH_SIZE=512, GAMMA=0.99, TAU=1e-2, LR=1e-4, UPDATE_EVERY=4, LOSS=F.mse_loss),
    "LunarLander": Config(DDQN=True, BUFFER_SIZE=int(7e5), BATCH_SIZE=512, GAMMA=0.99, TAU=1e-3, LR=3e-4, UPDATE_EVERY=4, LOSS=F.smooth_l1_loss),
    "Taxi": Config(DDQN=True, BUFFER_SIZE=int(5e5), BATCH_SIZE=512, GAMMA=0.8, TAU=1e-2, LR=5e-3, UPDATE_EVERY=4, LOSS=F.mse_loss)
}

def parse_args():
    parser = argparse.ArgumentParser(description="Script to load models for different environments.")

    parser.add_argument('--env_name', type=str, nargs='+', default=['CartPole-v1', 'LunarLander-v2', 'Taxi-v3'],
                        help='Name of the environment. Defaults to CartPole-v1, LunarLander-v2, and Taxi-v3.')

    args = parser.parse_args()

    if not isinstance(args.env_name, list):
        args.env_name = [args.env_name]
    
    args.model_path = [f'models/checkpoint_{env}.pth' for env in args.env_name]

    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for env_name, model_path in zip(args.env_name, args.model_path):
        print(f"Testing {env_name}")
        env, state_size, action_size = build_env(env_name)
        config = select_config(env_name, agent_configs)

        teacher = load_teacher(env_name, state_size, action_size)
        test_scores = test_model(teacher, env)
        # print(f"Test Scores over {len(test_scores)} runs in {env_name}: {test_scores}")
        print(f"Average Score: {np.mean(test_scores)}")
