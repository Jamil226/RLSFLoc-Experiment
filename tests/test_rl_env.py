import pytest
import numpy as np
import pandas as pd
import gymnasium as gym

from src.environment.rl_env_gymnasium import RLSFLocEnv

def test_rlsfloc_env_basic_and_reward_math():
    """
    Test RLSFLocEnv Custom Gymnasium environment.
    
    Verifies:
      - Observation and action spaces configurations
      - Context state (9D observation) is correct
      - Continuous actions softmax project onto valid 3D simplex correctly
      - Reward is positive for ranking improvement, zero for no change, and negative for degradation
    """
    # 1. Setup mock codebase metrics for 1 bug
    # 5 statements: s0, s1, s2, s3, s4
    # Ground truth fault is s3.
    # under raw execution (exec_norm), s3 is rank 2 (s4 is rank 1)
    df = pd.DataFrame([
        {'statement_id': 's0', 'exec_norm': 0.1, 'struct_norm': 0.1, 'semantic_norm': 0.1},
        {'statement_id': 's1', 'exec_norm': 0.2, 'struct_norm': 0.2, 'semantic_norm': 0.2},
        {'statement_id': 's2', 'exec_norm': 0.5, 'struct_norm': 0.5, 'semantic_norm': 0.5},
        {'statement_id': 's3', 'exec_norm': 0.6, 'struct_norm': 0.8, 'semantic_norm': 0.9}, # faulty
        {'statement_id': 's4', 'exec_norm': 0.9, 'struct_norm': 0.4, 'semantic_norm': 0.3}
    ])
    
    normalized_scores_list = [df]
    ground_truth_list = [["s3"]]
    
    # Instantiate environment
    env = RLSFLocEnv(
        normalized_scores_list,
        ground_truth_list,
        k=3, # Top-3 accuracy
        reward_weights={"top_k": 1.0, "mrr": 2.0, "exam": 5.0}
    )
    
    # 2. Test Reset
    obs, info = env.reset()
    assert obs.shape == (9,)
    assert info["num_statements"] == 5
    assert info["num_faults"] == 1
    
    # Check observation metrics (exec, struct, sem stats)
    # mean_exec = (0.1+0.2+0.5+0.6+0.9)/5 = 2.3/5 = 0.46
    assert np.isclose(obs[0], 0.46)
    # max_exec = 0.9
    assert np.isclose(obs[2], 0.9)
    # max_sem = 0.9
    assert np.isclose(obs[8], 0.9)
    
    # 3. Test Step with Improving Action
    # Action [0.0, 5.0, 5.0] puts all weight on structural and semantic (0.0, 0.5, 0.5)
    # This pushes s3's fused score to 0.85 and s4 to 0.35, ranking s3 to rank 1 (from baseline rank 2).
    action_improving = np.array([0.0, 10.0, 10.0]) # high values map to practically (0, 0.5, 0.5)
    obs2, reward, terminated, truncated, info2 = env.step(action_improving)
    
    assert terminated
    assert not truncated
    assert info2["lambda1"] < 0.05
    assert np.isclose(info2["lambda2"], 0.5, atol=0.05)
    assert np.isclose(info2["lambda3"], 0.5, atol=0.05)
    
    # Check ranks in info
    assert info2["avg_baseline_rank"] == 2.0
    assert info2["avg_fused_rank"] == 1.0 # Improved!
    
    # Check improvements
    # Top-k: baseline r=2 (<=3) is 1.0, fused r=1 (<=3) is 1.0 -> improvement is 0.0
    assert info2["top_k_improvement"] == 0.0
    # MRR: 1/1 - 1/2 = 0.5
    assert info2["mrr_improvement"] == 0.5
    # EXAM: 2/5 - 1/5 = 0.2
    assert info2["exam_reduction"] == 0.2
    
    # Reward: 1.0 * 0.0 + 2.0 * 0.5 + 5.0 * 0.2 = 1.0 + 1.0 = 2.0
    assert np.isclose(reward, 2.0, atol=0.1)


def test_rlsfloc_env_simplex_constraints():
    """
    Test that RLSFLocEnv projects arbitrary action outputs (including negative values)
    perfectly onto a valid 3D simplex.
    """
    df = pd.DataFrame([
        {'statement_id': 's0', 'exec_norm': 0.5, 'struct_norm': 0.5, 'semantic_norm': 0.5}
    ])
    env = RLSFLocEnv([df], [["s0"]])
    env.reset()
    
    # A. Arbitrary action with negative and high values
    action = np.array([-10.0, 5.0, 2.0])
    _, _, _, _, info = env.step(action)
    
    l1, l2, l3 = info["lambda1"], info["lambda2"], info["lambda3"]
    
    # Assert constraints
    assert l1 >= 0.0 and l2 >= 0.0 and l3 >= 0.0
    assert np.isclose(l1 + l2 + l3, 1.0)
    # l2 should be the highest since 5.0 is the highest action value
    assert l2 > l3 > l1
