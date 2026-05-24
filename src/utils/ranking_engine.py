import pandas as pd
import numpy as np
from src.utils.fusion import fuse_suspiciousness_scores

def generate_fault_ranking_report(bug_data, lambda1=0.3787, lambda2=0.3245, lambda3=0.2969):
    """
    Builds the final RLSFLoc fault ranking report for a dataset of bugs.
    Fuses metrics using learned weights, assigns robust ranks (handling ties via min-ranking),
    and filters to actual faulty statements to determine their localized ranks.
    
    Parameters:
    -----------
    bug_data : dict
        A dictionary mapping bug_id (e.g. 'defects4j-3') to a dictionary with keys:
          - 'scores': pandas.DataFrame containing ['statement_id', 'exec_norm', 'struct_norm', 'semantic_norm']
          - 'ground_truth': set or list of actual faulty statement IDs.
    lambda1 : float (default: 0.3787)
        Execution suspiciousness weight.
    lambda2 : float (default: 0.3245)
        Structural propagation weight.
    lambda3 : float (default: 0.2969)
        Semantic similarity weight.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns ['bug_id', 'statement_id', 'rank'] sorted by bug_id and rank.
    """
    rows = []
    
    for bug_id, data in bug_data.items():
        if 'scores' not in data or 'ground_truth' not in data:
            raise KeyError(f"Bug entry '{bug_id}' must contain both 'scores' and 'ground_truth' keys")
            
        scores_df = data['scores']
        ground_truth = set(data['ground_truth'])
        
        if scores_df.empty:
            continue
            
        # 1. Fuse scores and sort descending
        fused = fuse_suspiciousness_scores(
            scores_df, 
            lambda1=lambda1, 
            lambda2=lambda2, 
            lambda3=lambda3
        )
        
        # 2. Assign robust 1-based ranks handling ties (method='min')
        # Min rank gives tied elements the same lowest rank (e.g. if two items tie for 1st, both get rank 1)
        fused['rank'] = fused['fusion_score'].rank(ascending=False, method='min')
        
        # 3. Match actual faulty statements
        for fault in ground_truth:
            match = fused[fused['statement_id'] == fault]
            if not match.empty:
                assigned_rank = int(match.iloc[0]['rank'])
            else:
                # If ground truth was missing from scores, assign max rank penalty
                assigned_rank = len(fused) + 1
                
            rows.append({
                'bug_id': bug_id,
                'statement_id': fault,
                'rank': assigned_rank
            })
            
    result_df = pd.DataFrame(rows, columns=['bug_id', 'statement_id', 'rank'])
    # Sort report by bug_id and rank
    result_df = result_df.sort_values(by=['bug_id', 'rank']).reset_index(drop=True)
    return result_df
