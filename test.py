import argparse
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import os
import itertools
from agent import Agent, Config
from train import select_config, build_env
from rl import get_rollout
from dqn import QNetwork
from distill_model import load_teacher, load_dt_policy

def test_model(policy, env, n_runs=200):
    scores = []
    for _ in range(n_runs):
        state = np.array(env.reset()[0])
        score = 0
        done = False
        while not done:
            action = int(policy.predict(np.array([state]))[0])
            state, reward, terminated, truncated, *extra_vars = env.step(action)
            score += reward
            if terminated or truncated:
                break
        scores.append(score)

    return scores

# for student decision trees
def extract_params_from_filenames(filename):
    tree_prefix = "dt_policy_"
    if filename.startswith(tree_prefix):
        info = filename.removeprefix(tree_prefix).rsplit('.', 1)[0]
        parts = info.split('_')

        env_name = parts[0]
        tree_type_parts = parts[1:-2]  # Exclude the first (env_name) and the last two parts (depth and reweight)
        tree_type = '_'.join(tree_type_parts)
        depth_part = parts[-2]  # 'depthX'
        reweight_part = parts[-1]  # 'reweightTrue' or 'reweightFalse'

        max_depth = int(depth_part.replace('depth', ''))
        is_reweight = reweight_part.endswith('True')
        
        return env_name, tree_type, max_depth, is_reweight

    return None

def _checkpoint_paths(args):
    for env in args.env_names:
        if env == "WindyCartPole-v1":
            env = "CartPole-v1"

        yield f'{args.model_directory}/checkpoint_{env}.pth'


def parse_args():
    parser = argparse.ArgumentParser(description="Script to load models for different environments.")
    parser.add_argument('--env_names', type=str, nargs='+', default=['CartPole-v1', 'LunarLander-v2', 'Taxi-v3'],
                    help='Name of the environment. Defaults to all envs.')
    parser.add_argument('--model_type', type=str, choices=['student', 'teacher', 'all'], required=True,
                        help='Type of the model: student or teacher.')
    parser.add_argument('--model_directory', type=str, required=False, default="models", help="Model directory")  
    args = parser.parse_args()

    teacher_paths = list(_checkpoint_paths(args))
    student_paths = [f'{args.model_directory}/trees/{file}' for file in os.listdir(f'{args.model_directory}/trees') 
                        if file.endswith('.pk') and file.startswith('dt_policy')]
        
    match args.model_type:
        case 'teacher':
            args.model_paths = teacher_paths
        case 'student':
            args.model_paths = student_paths
        case 'all':
            args.model_paths = teacher_paths + student_paths

    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for env_name in args.env_names:
        print(f"=== Testing env {env_name} ===")
        for model_path in args.model_paths:
            env, state_size, action_size = build_env(env_name)

            dirname = os.path.dirname(model_path)
            filename = os.path.basename(model_path)
            if filename.startswith("dt_policy"):
                model = load_dt_policy(dirname, filename) if env_name in filename else None
            else:
                model = load_teacher(env_name, state_size, action_size) 

            if model:
                test_scores = test_model(model, env)
            else:
                continue

            print(f"Average Score: {np.mean(test_scores)}")

    # Export results to a CSV file
    # with open(f'csv/dt_policy_results_{config.env_name}.csv', mode='w', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=['depth', 'is_reweight', 'tree_type', 'reward'])
    #     writer.writeheader()
    #     writer.writerows(results)
