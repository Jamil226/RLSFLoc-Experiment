import numpy as np
import pandas as pd
import networkx as nx

def propagate_structural_suspiciousness(G, execution_scores, alpha=0.5):
    """
    Propagates execution suspiciousness scores across the structural dependency graph G
    using the dependency-aware propagation equation defined in Section 3.4 of the research paper:

    S_struct(v_i) = (1 - alpha) * S_exec(v_i) + alpha * sum_{v_j in N(v_i)} (w_ij * S_exec(v_j))

    Parameters:
    -----------
    G : networkx.DiGraph
        The directed structural dependency graph. Contains File, Method, and Statement nodes.
    execution_scores : dict, pandas.Series, or pandas.DataFrame
        Initial execution-based suspiciousness scores (e.g. Ochiai, Tarantula, or DStar).
    alpha : float, optional (default=0.5)
        Propagation coefficient in range [0, 1]. Controls the influence of structural neighbors.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: ['statement_id', 'structural_score']
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in the range [0.0, 1.0], but got {alpha}")

    # 1. Normalize execution_scores input into a unified dictionary lookup
    scores_dict = {}
    if isinstance(execution_scores, pd.DataFrame):
        cols = list(execution_scores.columns)
        if 'statement_id' in cols:
            stmt_col = 'statement_id'
            score_col = [c for c in cols if c != stmt_col][0]
        else:
            stmt_col = cols[0]
            score_col = cols[1]
        scores_dict = dict(zip(execution_scores[stmt_col], execution_scores[score_col]))
    elif isinstance(execution_scores, pd.Series):
        scores_dict = execution_scores.to_dict()
    elif isinstance(execution_scores, dict):
        scores_dict = execution_scores
    else:
        raise TypeError("execution_scores must be a dictionary, Pandas Series, or Pandas DataFrame")

    structural_scores = {}

    # 2. Iterate through nodes and propagate suspiciousness
    for node in G.nodes():
        # Get execution suspiciousness (default to 0.0 if not executed/scored)
        s_exec = float(scores_dict.get(node, 0.0))
        
        # Get outgoing dependency edges (ignoring contains edges which represent parent containment)
        successors = [
            v for u, v, d in G.out_edges(node, data=True)
            if d.get('type') != 'contains'
        ]
        
        if not successors:
            # If a node has no structural dependencies, keep its original execution suspiciousness
            structural_scores[node] = s_exec
        else:
            # Sum up edge weights. If weights are not normalized, normalize on the fly
            total_weight = sum(G[node][v].get('weight', 1.0) for v in successors)
            if total_weight > 0:
                neighbor_sum = sum(
                    (G[node][v].get('weight', 1.0) / total_weight) * float(scores_dict.get(v, 0.0))
                    for v in successors
                )
                structural_scores[node] = (1.0 - alpha) * s_exec + alpha * neighbor_sum
            else:
                structural_scores[node] = s_exec

    # 3. Format as a clean Pandas DataFrame
    results_df = pd.DataFrame({
        'statement_id': list(structural_scores.keys()),
        'structural_score': list(structural_scores.values())
    })

    return results_df


def tune_propagation_alpha(G, execution_scores, ground_truth_faults, alpha_grid=None):
    """
    Finds the optimal propagation coefficient (alpha) using a grid search over
    ranking quality (Mean Rank and Mean Reciprocal Rank) of ground-truth faulty elements.

    Parameters:
    -----------
    G : networkx.DiGraph
        The directed structural dependency graph.
    execution_scores : dict, pandas.Series, or pandas.DataFrame
        Initial execution-based suspiciousness scores.
    ground_truth_faults : list or set
        List/set of actual faulty statement/node IDs to evaluate rankings against.
    alpha_grid : list, optional
        Grid of candidate alpha values to search. Defaults to [0.0, 0.1, ..., 1.0].

    Returns:
    --------
    float
        The optimal alpha value that minimizes average rank (maximizes suspiciousness ranking).
    dict
        Detailed logs mapping each alpha candidate to its average rank and MRR.
    """
    if alpha_grid is None:
        alpha_grid = list(np.round(np.arange(0.0, 1.1, 0.1), 2))

    faults_set = set(ground_truth_faults)
    if not faults_set:
        raise ValueError("ground_truth_faults cannot be empty")

    tuning_logs = {}
    best_alpha = 0.5
    best_avg_rank = float('inf')

    # Grid search
    for alpha in alpha_grid:
        # 1. Propagate scores
        res_df = propagate_structural_suspiciousness(G, execution_scores, alpha=alpha)
        
        # 2. Sort by structural score descending
        # Add rank column (1-indexed, method='min' represents the highest rank in case of ties)
        res_df['rank'] = res_df['structural_score'].rank(ascending=False, method='min')
        
        # 3. Find rank of each ground truth fault
        ranks = []
        reciprocal_ranks = []
        for fault in faults_set:
            if fault in res_df['statement_id'].values:
                fault_rank = res_df.loc[res_df['statement_id'] == fault, 'rank'].values[0]
                ranks.append(fault_rank)
                reciprocal_ranks.append(1.0 / fault_rank)
            else:
                # If a fault is missing from the graph, penalize it with max rank
                max_rank = len(res_df)
                ranks.append(max_rank)
                reciprocal_ranks.append(1.0 / max_rank)
                
        avg_rank = np.mean(ranks) if ranks else float('inf')
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        tuning_logs[alpha] = {
            'avg_rank': avg_rank,
            'mrr': mrr
        }
        
        # Select optimal alpha based on minimizing average rank
        # (If tied, pick the one with higher MRR)
        if avg_rank < best_avg_rank:
            best_avg_rank = avg_rank
            best_alpha = alpha
        elif avg_rank == best_avg_rank:
            if mrr > tuning_logs.get(best_alpha, {}).get('mrr', 0.0):
                best_alpha = alpha

    return best_alpha, tuning_logs
