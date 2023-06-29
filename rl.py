# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import logging
# from ..util.log import *

NONE = 0
ERROR = 1
WARN = 2
INFO = 3
VERBOSE = 4
DEBUG = 5

log = print

counter = 0

def get_rollout(env, policy, render):
    done = False
    obs = np.array(env.reset()[0])
    rollout = []

    while not done:
        # Render
        if render:
            env.unwrapped.render()

        # Action
        # global counter
        # counter += 1
        # print(counter)
        # print("Predicting...")
        act = policy.predict(np.array([obs]))[0]

        # Step
        next_obs, rew, terminated, truncated, info = env.step(act)
        done = terminated or truncated

        # Rollout (s, a, r)
        # print("Rollowt")
        rollout.append((obs, act, rew))

        # Update (and remove LazyFrames)
        obs = np.array(next_obs)

    return rollout

def get_rollouts(env, policy, render, n_batch_rollouts):
    rollouts = []
    for i in range(n_batch_rollouts):
        rollouts.extend(get_rollout(env, policy, render))
    return rollouts

def _sample(obss, acts, qs, max_pts, is_reweight):
    # Step 1: Compute probabilities
    ps = np.max(qs, axis=1) - np.min(qs, axis=1)
    ps = ps / np.sum(ps)

    # Step 2: Sample points
    if is_reweight:
        # According to p(s)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), p=ps)
    else:
        # Uniformly (without replacement)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), replace=False)    

    # Step 3: Obtain sampled indices
    return obss[idx], acts[idx], qs[idx]

def test_policy(env, policy, n_test_rollouts):
    print("-- Testing policy")
    cum_rew = 0.0
    for i in range(n_test_rollouts):
        student_trace = get_rollout(env, policy, False)
        cum_rew += sum((rew for _, _, rew in student_trace))
    return cum_rew / n_test_rollouts

def test_policy_full(env, policy, n_test_rollouts):
    print("-- Testing policy")
    rewards = []
    for i in range(n_test_rollouts):
        # print("Ep #:", i)
        student_trace = get_rollout(env, policy, False)
        ep_rew = sum((rew for _, _, rew in student_trace))
        rewards.append(ep_rew)

    return rewards

def identify_best_policy(env, policies, n_test_rollouts):
    log('Initial policy count: {}'.format(len(policies)))
    # cut policies by half on each iteration
    while len(policies) > 1:
        # Step 1: Sort policies by current estimated reward
        policies = sorted(policies, key=lambda entry: -entry[1])

        # Step 2: Prune second half of policies
        n_policies = int((len(policies) + 1)/2)
        log('Current policy count: {}'.format(n_policies))

        # Step 3: build new policies
        new_policies = []
        for i in range(n_policies):
            policy, rew = policies[i]
            new_rew = test_policy(env, policy, n_test_rollouts)
            new_policies.append((policy, new_rew - rew))
            log('Reward update: {} -> {}'.format(rew, new_rew))

        policies = new_policies

    if len(policies) != 1:
        raise Exception()

    return policies[0][0]

def _get_action_sequences_helper(trace, seq_len):
    acts = [act for _, act, _ in trace]
    seqs = []
    for i in range(len(acts) - seq_len + 1):
        seqs.append(acts[i:i+seq_len])
    return seqs

def get_action_sequences(env, policy, seq_len, n_rollouts):
    # Step 1: Get action sequences
    seqs = []
    for _ in range(n_rollouts):
        trace = get_rollout(env, policy, False)
        seqs.extend(_get_action_sequences_helper(trace, seq_len))

    # Step 2: Bin action sequences
    counter = {}
    for seq in seqs:
        s = str(seq)
        if s in counter:
            counter[s] += 1
        else:
            counter[s] = 1

    # Step 3: Sort action sequences
    seqs_sorted = sorted(list(counter.items()), key=lambda pair: -pair[1])

    return seqs_sorted

def train_dagger(env, teacher, student, max_iters, n_batch_rollouts, max_samples, train_frac, is_reweight, n_test_rollouts):
    # Step 0: Setup
    obss, acts, qs = [], [], []
    students = []
    
    # Step 1: Generate some supervised traces into the buffer
    trace = get_rollouts(env, teacher, False, n_batch_rollouts)
    obss.extend((obs for obs, _, _ in trace))
    acts.extend((act for _, act, _ in trace))
    qs.extend(teacher.predict_q(np.array([obs for obs, _, _ in trace])))

    # Step 2: Dagger outer loop
    for i in range(max_iters):
        log('Iteration {}/{}'.format(i, max_iters))

        # Step 2a: Train from a random subset of aggregated data
        cur_obss, cur_acts, cur_qs = _sample(np.array(obss), np.array(acts), np.array(qs), max_samples, is_reweight)
        log('Training student with {} points'.format(len(cur_obss)))
        student.train(cur_obss, cur_acts, train_frac)

        # Step 2b: Generate trace using student
        student_trace = get_rollouts(env, student, False, n_batch_rollouts)
        student_obss = [obs for obs, _, _ in student_trace]
        
        # Step 2c: Query the oracle for supervision
        teacher_qs = teacher.predict_q(student_obss) # at the interface level, order matters, since teacher.predict may run updates
        teacher_acts = teacher.predict(student_obss)

        # Step 2d: Add the augmented state-action pairs back to aggregate
        obss.extend((obs for obs in student_obss))
        acts.extend(teacher_acts)
        qs.extend(teacher_qs)

        # Step 2e: Estimate the reward
        cur_rew = sum((rew for _, _, rew in student_trace)) / n_batch_rollouts
        log('Student reward: {}'.format(cur_rew))

        students.append((student.clone(), cur_rew))


    max_student = identify_best_policy(env, students, n_test_rollouts)

    print("found the best student")
    return max_student