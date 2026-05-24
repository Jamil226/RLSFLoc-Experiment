import pytest
import networkx as nx
import numpy as np
import pandas as pd

from src.utils.propagation import propagate_structural_suspiciousness, tune_propagation_alpha

def test_propagation_correctness():
    """
    Test structural propagation mathematical correctness against hand-calculated values.
    
    Test setup:
      - 3 nodes: n0, n1, n2
      - Directed edges (type != 'contains'):
        n0 -> n1 with weight 0.4
        n0 -> n2 with weight 0.6
        n1 and n2 have no successors
      - Execution scores:
        n0: 0.1, n1: 0.5, n2: 0.9
      - alpha = 0.5
      
    Hand calculations:
      - n0 has successors.
        S_struct(n0) = (1 - 0.5) * 0.1 + 0.5 * (0.4 * 0.5 + 0.6 * 0.9)
                     = 0.05 + 0.5 * (0.2 + 0.54) = 0.05 + 0.37 = 0.42
      - n1 has no successors.
        S_struct(n1) = S_exec(n1) = 0.5
      - n2 has no successors.
        S_struct(n2) = S_exec(n2) = 0.9
    """
    G = nx.DiGraph()
    G.add_node("n0", type="statement")
    G.add_node("n1", type="statement")
    G.add_node("n2", type="statement")
    
    # Add dependency edges (type != 'contains')
    G.add_edge("n0", "n1", type="control_flow", weight=0.4)
    G.add_edge("n0", "n2", type="data_flow", weight=0.6)
    
    exec_scores = {"n0": 0.1, "n1": 0.5, "n2": 0.9}
    
    # Run propagation
    res = propagate_structural_suspiciousness(G, exec_scores, alpha=0.5)
    
    # Verify results
    assert len(res) == 3
    
    n0_score = res.loc[res['statement_id'] == 'n0', 'structural_score'].values[0]
    n1_score = res.loc[res['statement_id'] == 'n1', 'structural_score'].values[0]
    n2_score = res.loc[res['statement_id'] == 'n2', 'structural_score'].values[0]
    
    assert np.isclose(n0_score, 0.42)
    assert np.isclose(n1_score, 0.5)
    assert np.isclose(n2_score, 0.9)


def test_propagation_alpha_zero():
    """
    Verify that if alpha = 0.0, structural scores are identical to execution scores.
    """
    G = nx.DiGraph()
    G.add_edge("a", "b", type="call", weight=0.5)
    
    exec_scores = {"a": 0.35, "b": 0.75}
    
    res = propagate_structural_suspiciousness(G, exec_scores, alpha=0.0)
    
    a_score = res.loc[res['statement_id'] == 'a', 'structural_score'].values[0]
    b_score = res.loc[res['statement_id'] == 'b', 'structural_score'].values[0]
    
    assert np.isclose(a_score, 0.35)
    assert np.isclose(b_score, 0.75)


def test_propagation_tuning():
    """
    Verify that tune_propagation_alpha correctly finds the optimal alpha
    minimizing the rank of the faulty statement.
    
    Test setup (same as test_propagation_correctness):
      - n0: execution = 0.1, n1: 0.5, n2: 0.9
      - n0 -> n1 (0.4), n0 -> n2 (0.6)
      - Faulty node is n0.
      
    Ranks of n0:
      - alpha = 0.0: scores are n0:0.1, n1:0.5, n2:0.9 -> rank of n0 is 3 (minimum rank)
      - alpha = 0.5: scores are n0:0.42, n1:0.5, n2:0.9 -> rank of n0 is 3
      - alpha = 1.0: S_struct(n0) = 0.74 -> scores are n0:0.74, n1:0.5, n2:0.9 -> rank of n0 is 2
      
    Therefore, alpha = 1.0 is the best because it minimizes the rank of n0 to 2 (instead of 3).
    """
    G = nx.DiGraph()
    G.add_node("n0")
    G.add_node("n1")
    G.add_node("n2")
    G.add_edge("n0", "n1", type="control_flow", weight=0.4)
    G.add_edge("n0", "n2", type="data_flow", weight=0.6)
    
    exec_scores = {"n0": 0.1, "n1": 0.5, "n2": 0.9}
    ground_truth = ["n0"]
    
    best_alpha, logs = tune_propagation_alpha(
        G,
        exec_scores,
        ground_truth,
        alpha_grid=[0.0, 0.5, 1.0]
    )
    
    assert best_alpha == 1.0
    assert logs[0.0]['avg_rank'] == 3.0
    assert logs[0.5]['avg_rank'] == 3.0
    assert logs[1.0]['avg_rank'] == 2.0
    assert logs[1.0]['mrr'] == 0.5  # 1.0 / 2.0 = 0.5
