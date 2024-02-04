import argparse
import csv
import torch
import numpy as np
import os
import time
from train import build_env
from distill_model import load_teacher, load_dt_policy


def test_model(policy, env, n_runs=20):
    scores = []
    times = []

    for _ in range(n_runs):
        state = np.array(env.reset()[0])
        score = 0
        done = False

        while not done:
            start_time = time.perf_counter_ns()
            action = int(policy.predict(np.array([state]))[0])
            end_time = time.perf_counter_ns()

            prediction_time = (end_time - start_time) / 1_000
            times.append(prediction_time)

            state, reward, terminated, truncated, _info = env.step(action)
            score += reward
            if terminated or truncated:
                break

        scores.append(score)

    average_score = np.mean(scores)
    average_time = np.mean(times)

    return average_score, average_time


def extract_params_from_filename(filename):
    tree_prefix = "dt_policy_"
    if filename.startswith(tree_prefix):
        info = filename.removeprefix(tree_prefix).rsplit('.', 1)[0]
        parts = info.split('_')

        # env_name = parts[0]
        # Exclude the first (env_name) and the last two parts (depth and reweight)
        tree_type_parts = parts[1:-2]
        tree_type = '_'.join(tree_type_parts)
        depth_part = parts[-2]  # 'depthX'
        reweight_part = parts[-1]  # 'reweightTrue' or 'reweightFalse'

        max_depth = int(depth_part.replace('depth', ''))
        is_reweight = reweight_part.endswith('True')

        return tree_type, max_depth, is_reweight

    return None, None, None


def _teacher_paths(args):
    # Initialize a dictionary to hold the paths for each environment
    env_paths = {env: [] for env in args.env_names}

    for env in args.env_names:
        mapped_env = "CartPole-v1" if env == "WindyCartPole-v1" else env
        path = f'{args.model_directory}/checkpoint_{mapped_env}.pth'
        env_paths[env].append(path)

    return env_paths


def _student_paths(args):
    # Initialize a dictionary to hold the paths for each environment
    env_paths = {env: ["models/trees/linear_dt_policy.pk"] for env in args.env_names}
    # env_paths = {env: [] for env in args.env_names}

    # for env in args.env_names:
    #     mapped_env = "CartPole-v1" if env == "WindyCartPole-v1" else env
    #     for file in os.listdir(f'{args.model_directory}/trees'):
    #         if file.endswith('.pk') and file.startswith('dt_policy') and mapped_env in file:
    #             full_path = f'{args.model_directory}/trees/{file}'
    #             env_paths[env].append(full_path)

    return env_paths

def _merge_paths(teacher_paths, student_paths):
    return {env: [teacher_path] + student_paths.get(env, [])
            for env, teacher_path in teacher_paths.items()}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to load models for different environments.")
    parser.add_argument('--env_names', type=str, nargs='+', default=['Taxi-v3'],
                        help='Name of the environment. Defaults to all envs.')
    parser.add_argument('--model_type', type=str, choices=['student', 'teacher', 'all'], required=True,
                        help='Type of the model: student or teacher.')
    parser.add_argument('--model_directory', type=str,
                        required=False, default="models", help="Model directory")
    args = parser.parse_args()

    teacher_paths = _teacher_paths(args)
    student_paths = _student_paths(args)

    match args.model_type:
        case 'teacher':
            args.model_paths = teacher_paths
        case 'student':
            args.model_paths = student_paths
        case 'all':
            args.model_paths = _merge_paths(teacher_paths, student_paths)

    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for env_name in args.env_names:
        rows = []
        for model_path in args.model_paths[env_name]:
            print("-- Path:", model_path)
            env, state_size, action_size = build_env(env_name)

            dirname = os.path.dirname(model_path)
            filename = os.path.basename(model_path)
            if filename.startswith("dt_policy") or filename.startswith("linear_dt_policy"):
                model = load_dt_policy(dirname, filename)
                model_type = "NA"
            else:
                model = load_teacher(model_path, state_size, action_size)
                model_type = "DDQN"

            print(f"- {model_path} -")
            scores, times = test_model(model, env)

            # model_type, depth, is_reweight, env, scores, task
            # row.append([filename, soce])
            tree_type, max_depth, is_reweight = extract_params_from_filename(filename)
            if tree_type:
                model_type = tree_type

            # ["model_type", "max_depth", "is_reweight", "avg_reward", "avg_time"]
            rows.append([model_type, max_depth, is_reweight, scores, times])
            print(f"Average Score: {scores}")
            print(f"Average Prediction Time: {times}")

        # Export results to a CSV file
        with open(f'output.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["model_type", "max_depth", "is_reweight", "avg_reward", "avg_time"])
            writer.writerows(rows)
