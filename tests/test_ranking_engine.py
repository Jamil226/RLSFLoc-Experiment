import pytest
import pandas as pd
import numpy as np
from src.utils.ranking_engine import generate_fault_ranking_report

def test_fault_ranking_report_basic():
    """
    Test generate_fault_ranking_report constructs report correctly for multiple bugs
    and returns exact matching ranks.
    """
    # 2 mock bugs
    # bug1 has 3 statements, fault is s1 (which has high structural and semantic scores).
    # Custom weights [0.5, 0.3, 0.2] gives:
    # s0: 0.5*0.1 + 0.3*0.1 + 0.2*0.1 = 0.10
    # s1: 0.5*0.4 + 0.3*0.9 + 0.2*0.8 = 0.20 + 0.27 + 0.16 = 0.63 (faulty)
    # s2: 0.5*0.9 + 0.3*0.2 + 0.2*0.1 = 0.45 + 0.06 + 0.02 = 0.53
    # Ranks: s1=1, s2=2, s0=3
    
    # bug2 has 2 statements, fault is s4 (which has low execution, but very high structural/semantic scores).
    # s3: 0.5*0.8 + 0.3*0.1 + 0.2*0.1 = 0.40 + 0.03 + 0.02 = 0.45
    # s4: 0.5*0.2 + 0.3*0.9 + 0.2*0.9 = 0.10 + 0.27 + 0.18 = 0.55 (faulty)
    # Ranks: s4=1, s3=2
    
    bug_data = {
        'bug1': {
            'scores': pd.DataFrame([
                {'statement_id': 's0', 'exec_norm': 0.1, 'struct_norm': 0.1, 'semantic_norm': 0.1},
                {'statement_id': 's1', 'exec_norm': 0.4, 'struct_norm': 0.9, 'semantic_norm': 0.8}, # faulty
                {'statement_id': 's2', 'exec_norm': 0.9, 'struct_norm': 0.2, 'semantic_norm': 0.1}
            ]),
            'ground_truth': ['s1']
        },
        'bug2': {
            'scores': pd.DataFrame([
                {'statement_id': 's3', 'exec_norm': 0.8, 'struct_norm': 0.1, 'semantic_norm': 0.1},
                {'statement_id': 's4', 'exec_norm': 0.2, 'struct_norm': 0.9, 'semantic_norm': 0.9}  # faulty
            ]),
            'ground_truth': ['s4']
        }
    }
    
    report = generate_fault_ranking_report(bug_data, lambda1=0.5, lambda2=0.3, lambda3=0.2)
    
    assert report.shape == (2, 3)
    assert list(report.columns) == ['bug_id', 'statement_id', 'rank']
    
    # Verify bug1 fault rank
    row_bug1 = report[report['bug_id'] == 'bug1'].iloc[0]
    assert row_bug1['statement_id'] == 's1'
    assert row_bug1['rank'] == 1
    
    # Verify bug2 fault rank
    row_bug2 = report[report['bug_id'] == 'bug2'].iloc[0]
    assert row_bug2['statement_id'] == 's4'
    assert row_bug2['rank'] == 1

def test_fault_ranking_report_missing_fault():
    """
    Verify that if a faulty statement is missing from the codebase scores,
    it is gracefully assigned a max rank penalty of len(fused) + 1.
    """
    bug_data = {
        'bug1': {
            'scores': pd.DataFrame([
                {'statement_id': 's0', 'exec_norm': 0.5, 'struct_norm': 0.5, 'semantic_norm': 0.5}
            ]),
            'ground_truth': ['missing_fault']
        }
    }
    
    report = generate_fault_ranking_report(bug_data)
    
    assert report.shape == (1, 3)
    assert report.iloc[0]['statement_id'] == 'missing_fault'
    assert report.iloc[0]['rank'] == 2 # len(fused) + 1 = 1 + 1 = 2

def test_fault_ranking_report_validation():
    """
    Test input schema verification.
    """
    bug_data_invalid = {
        'bug1': {
            'scores': pd.DataFrame([])
        }
    }
    with pytest.raises(KeyError):
        generate_fault_ranking_report(bug_data_invalid)
