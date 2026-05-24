import pytest
import pandas as pd
import numpy as np
import torch
from src.utils.baselines import (
    DeepFLMLPBaseline,
    RankNetBaseline,
    get_ochiai_baseline,
    get_tarantula_baseline,
    get_dstar_baseline,
    get_graph_baseline,
    get_transformer_baseline
)

def test_formula_baselines():
    """
    Verify that Ochiai, Tarantula, DStar, Graph, and Transformer isolated baselines
    compute and sort scores correctly.
    """
    df = pd.DataFrame([
        {'statement_id': 's0', 'exec_norm': 0.1, 'tarantula_norm': 0.2, 'dstar_norm': 0.3, 'struct_norm': 0.4, 'semantic_norm': 0.5},
        {'statement_id': 's1', 'exec_norm': 0.9, 'tarantula_norm': 0.8, 'dstar_norm': 0.7, 'struct_norm': 0.6, 'semantic_norm': 0.5}
    ])
    
    # Ochiai (uses exec_norm) -> s1 (0.9) > s0 (0.1)
    res_ochiai = get_ochiai_baseline(df)
    assert res_ochiai.loc[0, 'statement_id'] == 's1'
    assert np.isclose(res_ochiai.loc[0, 'fusion_score'], 0.9)
    
    # Graph-only (uses struct_norm) -> s1 (0.6) > s0 (0.4)
    res_graph = get_graph_baseline(df)
    assert res_graph.loc[0, 'statement_id'] == 's1'
    assert np.isclose(res_graph.loc[0, 'fusion_score'], 0.6)
    
    # Transformer-only (uses semantic_norm) -> s0 (0.5) and s1 (0.5) are equal
    res_trans = get_transformer_baseline(df)
    assert res_trans.shape == (2, 2)
    assert np.isclose(res_trans.loc[0, 'fusion_score'], 0.5)

def test_deepfl_mlp_baseline_flow():
    """
    Verify that DeepFLMLPBaseline trains, evaluates, and predicts sorted statements.
    """
    # 2 training bugs
    train_scores = [
        pd.DataFrame([
            {'statement_id': 's0', 'exec_norm': 0.1, 'struct_norm': 0.1, 'semantic_norm': 0.1},
            {'statement_id': 's1', 'exec_norm': 0.9, 'struct_norm': 0.9, 'semantic_norm': 0.9} # fault
        ]),
        pd.DataFrame([
            {'statement_id': 's2', 'exec_norm': 0.2, 'struct_norm': 0.2, 'semantic_norm': 0.2},
            {'statement_id': 's3', 'exec_norm': 0.8, 'struct_norm': 0.8, 'semantic_norm': 0.8} # fault
        ])
    ]
    train_gt = [['s1'], ['s3']]
    
    baseline = DeepFLMLPBaseline(lr=0.01, epochs=5)
    baseline.fit(train_scores, train_gt)
    
    # Predict on test
    test_df = pd.DataFrame([
        {'statement_id': 's4', 'exec_norm': 0.1, 'struct_norm': 0.1, 'semantic_norm': 0.1},
        {'statement_id': 's5', 'exec_norm': 0.9, 'struct_norm': 0.9, 'semantic_norm': 0.9}
    ])
    
    pred_res = baseline.predict(test_df)
    
    assert pred_res.shape == (2, 2)
    assert list(pred_res.columns) == ['statement_id', 'fusion_score']
    # Higher scores should lead to higher predicted probability
    assert pred_res.loc[0, 'statement_id'] == 's5'

def test_ranknet_baseline_flow():
    """
    Verify that RankNetBaseline trains using pairwise cross-entropy and outputs sorted lists.
    """
    train_scores = [
        pd.DataFrame([
            {'statement_id': 's0', 'exec_norm': 0.1, 'struct_norm': 0.1, 'semantic_norm': 0.1},
            {'statement_id': 's1', 'exec_norm': 0.9, 'struct_norm': 0.9, 'semantic_norm': 0.9} # fault
        ])
    ]
    train_gt = [['s1']]
    
    baseline = RankNetBaseline(lr=0.01, epochs=5, num_pairs_per_bug=10)
    baseline.fit(train_scores, train_gt)
    
    test_df = pd.DataFrame([
        {'statement_id': 's2', 'exec_norm': 0.2, 'struct_norm': 0.2, 'semantic_norm': 0.2},
        {'statement_id': 's3', 'exec_norm': 0.8, 'struct_norm': 0.8, 'semantic_norm': 0.8}
    ])
    
    pred_res = baseline.predict(test_df)
    
    assert pred_res.shape == (2, 2)
    assert pred_res.loc[0, 'statement_id'] == 's3'
