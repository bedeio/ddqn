from typing import Any, Mapping
import torch
import torch.nn as nn

import torch.nn.functional as F


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

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
    
    def predict(self, obss):
        def one():
            for next_states in obss:
                next_states = torch.from_numpy(next_states).float()
                qs = self.forward(next_states)
                _, indices = torch.sort(qs, descending=True)
                yield indices[0].numpy()

        with torch.no_grad():
            return list(one())

    def predict_q(self, obss):
        def one():
            for next_states in obss:
                next_states = torch.from_numpy(next_states).float()
                yield self.forward(next_states).numpy()

        with torch.no_grad():
            return list(one())
        


