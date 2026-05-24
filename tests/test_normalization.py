import pytest
import pandas as pd
import numpy as np

from src.utils.normalization import normalize_rlsfloc_scores

def test_multi_score_normalization_correctness():
    """
    Test multi-score normalization correctness against expected hand-calculated bounds.
    
    Test setup:
      - 2 statements: s0, s1
      - Raw scores:
        exec: s0=2.0, s1=4.0 -> min=2.0, max=4.0
        struct: s0=10.0, s1=20.0 -> min=10.0, max=20.0
        semantic: s0=0.3, s1=0.7 -> min=0.3, max=0.7
        
    Expected normalized outcomes:
      - s0 (minimums):
        exec_norm = (2.0 - 2.0)/(2.0 + 1e-12) = 0.0
        struct_norm = (10.0 - 10.0)/(10.0 + 1e-12) = 0.0
        semantic_norm = (0.3 - 0.3)/(0.4 + 1e-12) = 0.0
      - s1 (maximums):
        exec_norm = (4.0 - 2.0)/(2.0 + 1e-12) = 1.0
        struct_norm = (20.0 - 10.0)/(10.0 + 1e-12) = 1.0
        semantic_norm = (0.7 - 0.3)/(0.4 + 1e-12) = 1.0
    """
    exec_scores = {"s0": 2.0, "s1": 4.0}
    struct_scores = {"s0": 10.0, "s1": 20.0}
    sem_scores = {"s0": 0.3, "s1": 0.7}
    
    df = normalize_rlsfloc_scores(exec_scores, struct_scores, sem_scores, epsilon=1e-12)
    
    # Verify outputs
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['statement_id', 'exec_norm', 'struct_norm', 'semantic_norm']
    assert len(df) == 2
    
    # Rows should be sorted by statement_id ('s0', 's1')
    assert list(df['statement_id']) == ['s0', 's1']
    
    # s0 row assertions
    s0_row = df.iloc[0]
    assert np.isclose(s0_row['exec_norm'], 0.0)
    assert np.isclose(s0_row['struct_norm'], 0.0)
    assert np.isclose(s0_row['semantic_norm'], 0.0)
    
    # s1 row assertions
    s1_row = df.iloc[1]
    assert np.isclose(s1_row['exec_norm'], 1.0)
    assert np.isclose(s1_row['struct_norm'], 1.0)
    assert np.isclose(s1_row['semantic_norm'], 1.0)


def test_normalization_input_format_compatibility():
    """
    Verify that normalize_rlsfloc_scores successfully processes mixed input formats:
      - execution_scores: Pandas DataFrame
      - structural_scores: Pandas Series
      - semantic_scores: Python Dictionary
    """
    exec_df = pd.DataFrame({
        'statement_id': ['s1', 's2', 's3'],
        'Ochiai': [0.1, 0.5, 0.9]
    })
    
    struct_series = pd.Series([10, 50, 90], index=['s1', 's2', 's3'])
    
    sem_dict = {'s1': 0.0, 's2': 0.5, 's3': 1.0}
    
    df = normalize_rlsfloc_scores(exec_df, struct_series, sem_dict, epsilon=1e-12)
    
    assert len(df) == 3
    assert list(df['statement_id']) == ['s1', 's2', 's3']
    
    # Min elements (s1)
    s1_row = df.iloc[0]
    assert np.isclose(s1_row['exec_norm'], 0.0)
    assert np.isclose(s1_row['struct_norm'], 0.0)
    assert np.isclose(s1_row['semantic_norm'], 0.0)
    
    # Max elements (s3)
    s3_row = df.iloc[2]
    assert np.isclose(s3_row['exec_norm'], 1.0)
    assert np.isclose(s3_row['struct_norm'], 1.0)
    assert np.isclose(s3_row['semantic_norm'], 1.0)
    
    # Mid elements (s2)
    s2_row = df.iloc[1]
    assert np.isclose(s2_row['exec_norm'], 0.5)
    assert np.isclose(s2_row['struct_norm'], 0.5)
    assert np.isclose(s2_row['semantic_norm'], 0.5)
    
    print("\n[Verification Log] Multi-Score Normalization tests passed.")
