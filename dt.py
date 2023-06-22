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

import os
import pickle as pk
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from lineartree import LinearTreeClassifier

log = print

def accuracy(policy, obss, acts):
    return np.mean(acts == policy.predict(obss))

def split_train_test(obss, acts, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = obss[idx[:n_train]]
    acts_train = acts[idx[:n_train]]
    obss_test = obss[idx[n_train:]]
    acts_test = acts[idx[n_train:]]
    return obss_train, acts_train, obss_test, acts_test

def save_dt_policy(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    f = open(dirname + '/' + fname, 'wb')
    pk.dump(dt_policy, f)
    f.close()

def save_dt_policy_viz(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    if hasattr(dt_policy.tree, "tree_"):
        export_graphviz(dt_policy.tree, dirname + '/' + fname)
    else:
        feature_names = ["x", "y", "dx", "dy", "theta", "d_theta", "left_contact", "right_contact"]
        dot = dt_policy.tree.model_to_dot(feature_names=feature_names)
        dot.write(f'{dirname}/{fname}')
        dot.write_png(f'{dirname}/{fname}.png')
        

def load_dt_policy(dirname, fname):
    f = open(dirname + '/' + fname, 'rb')
    dt_policy = pk.load(f)
    f.close()
    return dt_policy

class DTPolicy:
    def __init__(self, max_depth):
        self.max_depth = max_depth
    
    def fit(self, obss, acts):
        self.tree = LinearTreeClassifier(max_depth=self.max_depth, base_estimator=RidgeClassifier(), criterion='hamming')
        # self.tree = LinearTreeClassifier(max_depth=self.max_depth, base_estimator=LogisticRegression(solver='saga'), criterion='crossentropy')
        # self.tree = DecisionTreeClassifier(max_depth=self.max_depth, ccp_alpha=0.001)
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
        self.fit(obss_train, acts_train)
        log('Train accuracy: {}'.format(accuracy(self, obss_train, acts_train)))
        log('Test accuracy: {}'.format(accuracy(self, obss_test, acts_test)))
        # log('Number of nodes: {}'.format(self.tree.tree_.node_count))

    def predict(self, obss):
        return self.tree.predict(obss)

    def clone(self):
        clone = DTPolicy(self.max_depth)
        clone.tree = self.tree
        return clone
