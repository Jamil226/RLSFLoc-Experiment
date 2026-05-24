import numpy as np
import gym
from gym import spaces

class FaultLocalizationEnv(gym.Env):
    """
    Custom Gym Environment for Reinforcement Learning-based Fault Localization.
    The agent steps through candidate components (observation space: software metrics or embeddings)
    and decides whether to recommend them for inspection (action space: 1 = rank highly, 0 = skip/rank lowly).
    """
    def __init__(self, candidates, labels):
        super(FaultLocalizationEnv, self).__init__()
        self.candidates = candidates  # Pandas DataFrame of features/metrics
        self.labels = labels          # Target labels (1 for buggy, 0 for clean)
        self.num_candidates = len(candidates)
        
        # Action space: 0 (keep low rank) or 1 (promote to top ranks)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: candidate metric values
        self.feature_dim = candidates.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32
        )
        self.current_idx = 0

    def reset(self):
        """
        Reset environment state to start a new ranking episode over the codebase.
        """
        self.current_idx = 0
        state = self.candidates.iloc[self.current_idx].values.astype(np.float32)
        return state

    def step(self, action):
        """
        Take an action on the current candidate and receive reward/next candidate.
        """
        done = False
        label = self.labels.iloc[self.current_idx]
        
        # Reward design
        # Reward correct positive localization; penalize false positives and false negatives
        if action == 1 and label == 1:
            reward = 2.0  # Big reward for successfully locating bug
        elif action == 1 and label == 0:
            reward = -0.5  # Penalty for false positive inspection cost
        elif action == 0 and label == 1:
            reward = -1.5  # Heavy penalty for missing a bug (false negative)
        else:
            reward = 0.1   # Small positive reward for correctly skipping clean components
            
        self.current_idx += 1
        if self.current_idx >= self.num_candidates:
            done = True
            state = np.zeros(self.feature_dim, dtype=np.float32)
        else:
            state = self.candidates.iloc[self.current_idx].values.astype(np.float32)
            
        return state, reward, done, {}
