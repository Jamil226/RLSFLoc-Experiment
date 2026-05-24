import pytest
import pandas as pd
import numpy as np
import time
from src.utils.metrics import evaluate_rlsfloc_performance

def test_evaluate_rlsfloc_performance_correctness():
    """
    Test that evaluate_rlsfloc_performance calculates standard and advanced metrics correctly
    over deterministic mock outputs.
    """
    # 2 mock bugs:
    # bug1 has 5 statements: s0, s1, s2, s3, s4. Faults: s2, s3.
    # ranked list: s1, s2, s0, s3, s4
    # fault ranks in list: s2 is index 1 -> rank 2, s3 is index 3 -> rank 4.
    # first matched fault rank: min(2, 4) = 2.
    # Reciprocal rank: 1/2 = 0.5
    # EXAM: 2/5 = 0.4
    # AP: faults at rank 2 and 4.
    #   rank 2: 1 fault. Precision = 1/2 = 0.5.
    #   rank 4: 2 faults. Precision = 2/4 = 0.5.
    #   AP = (0.5 + 0.5)/2 = 0.5.
    
    # bug2 has 4 statements: s0, s1, s2, s3. Fault: s0.
    # ranked list: s0, s2, s1, s3
    # fault rank: s0 is rank 1.
    # reciprocal rank: 1/1 = 1.0
    # EXAM: 1/4 = 0.25
    # AP: fault at rank 1. AP = 1/1 = 1.0.
    
    df1 = pd.DataFrame({'statement_id': ['s1', 's2', 's0', 's3', 's4']})
    df2 = pd.DataFrame({'statement_id': ['s0', 's2', 's1', 's3']})
    
    ranked_dfs = [df1, df2]
    ground_truths = [['s2', 's3'], ['s0']]
    
    metrics = evaluate_rlsfloc_performance(ranked_dfs, ground_truths)
    
    # Verify ranking metrics
    # Top-1 hits: bug2 (min_rank=1 <= 1) is a hit. bug1 (min_rank=2) is not. Rate = 1/2 = 0.5.
    assert metrics["top_1"] == 0.5
    # Top-3 hits: both are hits (min_rank 2 and 1 both <= 3). Rate = 2/2 = 1.0.
    assert metrics["top_3"] == 1.0
    assert metrics["top_5"] == 1.0
    assert metrics["top_10"] == 1.0
    
    # MRR = (0.5 + 1.0)/2 = 0.75
    assert np.isclose(metrics["mrr"], 0.75)
    
    # MAP = (0.5 + 1.0)/2 = 0.75
    assert np.isclose(metrics["map"], 0.75)
    
    # EXAM = (0.4 + 0.25)/2 = 0.325
    assert np.isclose(metrics["exam_score"], 0.325)

def test_evaluate_rlsfloc_performance_benchmarking():
    """
    Verify that providing an eval_func dynamically benchmarks runtime and peak memory.
    """
    df1 = pd.DataFrame({'statement_id': ['s0', 's1']})
    
    def mock_eval():
        # Add slight artificial delay to make sure time is non-zero
        time.sleep(0.05)
        # Allocate some small memory
        dummy = [x for x in range(100000)]
        return [df1]
        
    metrics = evaluate_rlsfloc_performance([], [['s0']], eval_func=mock_eval)
    
    # Verify benchmarking keys
    assert "runtime_sec" in metrics
    assert "peak_memory_mb" in metrics
    
    assert metrics["runtime_sec"] >= 0.04
    assert metrics["peak_memory_mb"] > 0.0

def test_evaluate_rlsfloc_performance_validation():
    """
    Verify dimension validations and exceptions.
    """
    with pytest.raises(ValueError):
        evaluate_rlsfloc_performance([], [])
        
    df = pd.DataFrame({'statement_id': ['s0']})
    with pytest.raises(ValueError):
        # Mismatch in lengths
        evaluate_rlsfloc_performance([df, df], [['s0']])
