import pytest
import pandas as pd
import numpy as np
from src.utils.fusion import fuse_suspiciousness_scores

def test_fuse_suspiciousness_scores_correctness():
    """
    Test that fuse_suspiciousness_scores computes correct linear combination
    and sorts the statements descending by fusion score.
    """
    df = pd.DataFrame([
        {'statement_id': 's0', 'exec_norm': 0.1, 'struct_norm': 0.2, 'semantic_norm': 0.3},
        {'statement_id': 's1', 'exec_norm': 0.9, 'struct_norm': 0.8, 'semantic_norm': 0.7},
        {'statement_id': 's2', 'exec_norm': 0.5, 'struct_norm': 0.5, 'semantic_norm': 0.5}
    ])
    
    # Custom weights: 0.5, 0.3, 0.2
    # s0: 0.5*0.1 + 0.3*0.2 + 0.2*0.3 = 0.05 + 0.06 + 0.06 = 0.17
    # s1: 0.5*0.9 + 0.3*0.8 + 0.2*0.7 = 0.45 + 0.24 + 0.14 = 0.83
    # s2: 0.5*0.5 + 0.3*0.5 + 0.2*0.5 = 0.50
    
    result = fuse_suspiciousness_scores(df, lambda1=0.5, lambda2=0.3, lambda3=0.2)
    
    # Verify shape and columns
    assert result.shape == (3, 2)
    assert list(result.columns) == ['statement_id', 'fusion_score']
    
    # Verify sorting order (descending by score: s1, s2, s0)
    assert result.loc[0, 'statement_id'] == 's1'
    assert np.isclose(result.loc[0, 'fusion_score'], 0.83)
    
    assert result.loc[1, 'statement_id'] == 's2'
    assert np.isclose(result.loc[1, 'fusion_score'], 0.50)
    
    assert result.loc[2, 'statement_id'] == 's0'
    assert np.isclose(result.loc[2, 'fusion_score'], 0.17)

def test_fuse_suspiciousness_scores_default_weights():
    """
    Verify that if weights are omitted, default PPO best learned weights
    are applied and computed correctly.
    """
    df = pd.DataFrame([
        {'statement_id': 's0', 'exec_norm': 1.0, 'struct_norm': 1.0, 'semantic_norm': 1.0}
    ])
    
    # Default weights are 0.3787, 0.3245, 0.2969 -> sum to 1.0
    result = fuse_suspiciousness_scores(df)
    
    assert np.isclose(result.loc[0, 'fusion_score'], 1.0)

def test_fuse_suspiciousness_scores_invalid_inputs():
    """
    Test validation errors for invalid input types and missing columns.
    """
    # 1. Invalid input type
    with pytest.raises(TypeError):
        fuse_suspiciousness_scores([1, 2, 3])
        
    # 2. Missing required columns
    df_invalid = pd.DataFrame([
        {'statement_id': 's0', 'exec_norm': 1.0}
    ])
    with pytest.raises(KeyError):
        fuse_suspiciousness_scores(df_invalid)
