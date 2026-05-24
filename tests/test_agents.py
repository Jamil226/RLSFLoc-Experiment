import pytest
import numpy as np
import torch
from src.agent.ppo import PPOAgent
from src.agent.actor_critic import ActorCriticAgent
from src.agent.dqn_discrete import SimplexDQNAgent, generate_simplex_grid

def test_simplex_grid_generator():
    """
    Verify that the simplex grid generator generates valid continuous weights
    on the 3D simplex.
    """
    grid = generate_simplex_grid(step_size=0.1)
    
    # Grid should contain 66 combinations for step_size=0.1
    assert len(grid) == 66
    
    # Each coordinate must sum to exactly 1.0 and be non-negative
    for weights in grid:
        assert weights.shape == (3,)
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0.0)

def test_ppo_agent_basic_flow():
    """
    Verify that PPOAgent handles action selection and backward optimization updates.
    """
    state_dim = 9
    action_dim = 3
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=0.01)
    
    # 1. Action selection
    state = np.random.rand(state_dim).astype(np.float32)
    action, log_prob, value = agent.select_action(state)
    
    assert action.shape == (3,)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    
    # 2. Update step
    states = [state, state]
    actions = [action, action]
    old_log_probs = [log_prob, log_prob]
    returns = [2.0, 1.0] # Direct rewards / returns
    
    loss = agent.update(states, actions, old_log_probs, returns)
    assert isinstance(loss, float)
    assert loss != 0.0

def test_actor_critic_agent_basic_flow():
    """
    Verify that ActorCriticAgent handles continuous action selection and td advantage updates.
    """
    state_dim = 9
    action_dim = 3
    agent = ActorCriticAgent(state_dim=state_dim, action_dim=action_dim, lr=0.01)
    
    # 1. Action selection
    state = np.random.rand(state_dim).astype(np.float32)
    action, log_prob, value = agent.select_action(state)
    
    assert action.shape == (3,)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    
    # 2. Update step
    states = [state, state]
    actions = [action, action]
    log_probs = [log_prob, log_prob]
    returns = [2.0, 1.0]
    
    loss = agent.update(states, actions, log_probs, returns)
    assert isinstance(loss, float)
    assert loss != 0.0

def test_simplex_dqn_agent_flow():
    """
    Verify that SimplexDQNAgent operates discrete epsilon-greedy updates and correctly
    applies the log-transform inversion.
    """
    state_dim = 9
    agent = SimplexDQNAgent(state_dim=state_dim, step_size=0.1, lr=0.01)
    
    # 1. Action selection
    state = np.random.rand(state_dim).astype(np.float32)
    action_idx, continuous_action, simplex_weights = agent.select_action(state, epsilon=0.0)
    
    assert 0 <= action_idx < agent.action_dim
    assert continuous_action.shape == (3,)
    assert simplex_weights.shape == (3,)
    assert np.isclose(np.sum(simplex_weights), 1.0)
    
    # Epsilon = 1.0 should select randomly
    action_idx_rand, _, _ = agent.select_action(state, epsilon=1.0)
    assert 0 <= action_idx_rand < agent.action_dim
    
    # 2. Log-transform inversion validation
    # Softmax of continuous_action should yield exactly simplex_weights!
    exp_act = np.exp(continuous_action - np.max(continuous_action))
    softmax_weights = exp_act / np.sum(exp_act)
    assert np.allclose(softmax_weights, simplex_weights)
    
    # 3. Training updates
    next_state = np.random.rand(state_dim).astype(np.float32)
    # Populate memory
    for _ in range(5):
        agent.remember(state, action_idx, 1.5, next_state, False)
        
    loss = agent.train_step(batch_size=4)
    assert isinstance(loss, float)
    assert loss > 0.0
