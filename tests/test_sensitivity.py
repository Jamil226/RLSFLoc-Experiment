import pytest
import pandas as pd
import numpy as np
from src.sensitivity_analysis import generate_sensitivity_bug_dataset

def test_generate_sensitivity_bug_dataset():
    """
    Test generate_sensitivity_bug_dataset correctly builds datasets with columns
    exec_norm, struct_norm, and semantic_norm.
    """
    scores, gt = generate_sensitivity_bug_dataset(num_bugs=3, alpha=0.4, seed=42)
    
    assert len(scores) == 3
    assert len(gt) == 3
    
    for df in scores:
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['statement_id', 'exec_norm', 'struct_norm', 'semantic_norm']
        assert df.shape[0] >= 30
        assert df.shape[0] <= 80
        
        # Values must be within [0.0, 1.0] due to min-max normalization
        assert df['exec_norm'].min() >= 0.0
        assert df['exec_norm'].max() <= 1.0
        assert df['struct_norm'].min() >= 0.0
        assert df['struct_norm'].max() <= 1.0
        assert df['semantic_norm'].min() >= 0.0
        assert df['semantic_norm'].max() <= 1.0
