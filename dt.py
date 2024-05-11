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
import pickle 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from lineartree import LinearTreeClassifier
from lightgbm import LGBMClassifier

log = print

def custom_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator

class BarebonesLogisticRegression(LogisticRegression):
    def predict(self, X):
        scores = np.dot(X, self.coef_.T) + self.intercept_
        if scores.shape[1] == 1:
            indices = (scores > 0).astype(int).flatten()
        else:
            indices = np.argmax(scores, axis=1)
        return self.classes_[indices]


def accuracy(policy, obss, acts):
    return np.mean(acts == policy.predict(obss))

def save_dt_policy(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    f = open(dirname + '/' + fname, 'wb')
    pickle.dump(dt_policy, f)
    f.close()

def save_dt_policy_viz(dt_policy, dirname, fname):
    feature_names = ["x", "y", "dx", "dy", "theta", "d_theta", "left_contact", "right_contact"]
    class_names = ["do nothing", "fire left engine", "fire main engine", "fire right engine"]

    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    if hasattr(dt_policy.tree, "tree_"):
        export_graphviz(dt_policy.tree, dirname + '/' + fname, feature_names=feature_names, class_names=class_names, filled=True)
        from subprocess import call
        call(['dot', '-Tpng', dirname + '/' + fname, '-o', dirname + '/dt_simple.png', '-Gdpi=300'])
    else:
        dot = dt_policy.tree.model_to_dot(feature_names=feature_names)
        dot.write(f'{dirname}/{fname}')
        dot.write_png(f'{dirname}/{fname}.png')
        

def load_dt_policy(dirname, fname):
    f = open(dirname + '/' + fname, 'rb')
    dt_policy = pickle.load(f)
    f.close()
    return dt_policy

class DTPolicy:
    def __init__(self, max_depth, tree_type):
        self.max_depth = max_depth
        self.tree_type = tree_type
        match tree_type:
            case 'decision_tree':
                self.tree = DecisionTreeClassifier(max_depth=self.max_depth, ccp_alpha=0.005)
            case 'linear_tree_logistic':
                self.tree = LinearTreeClassifier(max_depth=self.max_depth, base_estimator=BarebonesLogisticRegression(max_iter=350, solver='liblinear'), criterion='crossentropy')
            case 'linear_tree_ridge':
                self.tree = LinearTreeClassifier(max_depth=self.max_depth, base_estimator=RidgeClassifier(max_iter=250), criterion='hamming')
            case 'lightgbm_linear':
                self.tree = LGBMClassifier(n_estimators=5, num_leaves=31, verbosity=-1, objective="multiclass")
            case _:
                raise ValueError("Invalid tree type:", tree_type)
    
    def fit(self, obss, acts):
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, obss_test, acts_train, acts_test = train_test_split(obss, acts, train_size=train_frac)
        scaler = StandardScaler()

        # if self.tree_type == 'linear_tree_logistic':
        #     obss_train = scaler.fit_transform(obss_train)
        #     obss_test = scaler.transform(obss_test)

        self.fit(obss_train, acts_train)
        # log('Train accuracy: {}'.format(accuracy(self, obss_train_scaled, acts_train)))
        # log('Test accuracy: {}'.format(accuracy(self, obss_test_scaled, acts_test)))
        # log('Number of nodes: {}'.format(self.tree.tree_.node_count))

    def predict(self, obss):
        return self.tree.predict(obss)

    def clone(self):
        clone = DTPolicy(self.max_depth, self.tree_type)
        clone.tree = self.tree
        return clone
