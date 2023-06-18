log = print

import torch
import gymnasium as gym

from dqn import QNetwork
from rl import train_dagger, test_policy
from dt import DTPolicy, load_dt_policy, save_dt_policy, save_dt_policy_viz 

def load_model(q_network):
    q_network.load_state_dict(
        torch.load('checkpoint.pth', map_location=torch.device('cpu')))

def learn_dt():
    # Parameters
    log_fname = '../pong_dt.log'
    model_path = '../data/model-atari-pong-1/saved'
    max_depth = 12
    n_batch_rollouts = 10
    max_samples = 250000
    max_iters = 80
    train_frac = 0.8
    is_reweight = True
    n_test_rollouts = 50
    save_dirname = '/tmp/pong'
    save_fname = 'dt_policy.pk'
    save_viz_fname = 'dt_policy.dot'
    is_train = True
    
    # Logging
    # set_file(log_fname)
    
    # Data structures
    env = gym.make('LunarLander-v2')
    teacher = QNetwork(state_size=8, action_size=4, seed=42)
    load_model(teacher)

    student = DTPolicy(max_depth)

    # Train student
    if is_train:
        student = train_dagger(env, teacher, student, max_iters, n_batch_rollouts, max_samples, train_frac, is_reweight, n_test_rollouts)
        save_dt_policy(student, save_dirname, save_fname)
        save_dt_policy_viz(student, save_dirname, save_viz_fname)
    else:
        student = load_dt_policy(save_dirname, save_fname)

    # Test student
    rew = test_policy(env, student, n_test_rollouts)
    log('Final reward: {}'.format(rew))
    log('Number of nodes: {}'.format(student.tree.tree_.node_count))

def print_size():
    # Parameters
    dirname = 'results/run9'
    fname = 'dt_policy.pk'

    # Load decision tree
    dt = load_dt_policy(dirname, fname)

    # Size
    # print(dt.tree.tree_.node_count)

if __name__ == '__main__':
    learn_dt()