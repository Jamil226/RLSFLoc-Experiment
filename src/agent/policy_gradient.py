import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyGradientAgent:
    """
    REINFORCE / Policy Gradient Agent skeleton for RLSFLoc.
    """
    def __init__(self, state_dim, action_dim=2, lr=0.001):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        state_t = torch.FloatTensor(state)
        probs = self.policy(state_t)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
