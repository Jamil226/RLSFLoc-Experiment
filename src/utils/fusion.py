import pandas as pd
import numpy as np

def fuse_suspiciousness_scores(df_scores, lambda1=0.3787, lambda2=0.3245, lambda3=0.2969):
    """
    Perform final suspiciousness scores fusion for RLSFLoc:
    S_fusion = lambda1 * S_exec + lambda2 * S_struct + lambda3 * S_sem
    
    Parameters:
    -----------
    df_scores : pandas.DataFrame or dict
        A DataFrame containing columns: ['statement_id', 'exec_norm', 'struct_norm', 'semantic_norm'].
        Or a dictionary with these keys representing aligned scores.
    lambda1 : float (default: 0.3787, PPO optimal execution weight)
        Weight for execution-based normalization score (S_exec)
    lambda2 : float (default: 0.3245, PPO optimal structural weight)
        Weight for structural propagation normalization score (S_struct)
    lambda3 : float (default: 0.2969, PPO optimal semantic weight)
        Weight for semantic embedding similarity normalization score (S_sem)
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing ['statement_id', 'fusion_score'] sorted descending by fusion_score.
    """
    if isinstance(df_scores, dict):
        df = pd.DataFrame(df_scores)
    elif isinstance(df_scores, pd.DataFrame):
        df = df_scores.copy()
    else:
        raise TypeError("df_scores must be a pandas.DataFrame or a dictionary")
        
    required_cols = ['statement_id', 'exec_norm', 'struct_norm', 'semantic_norm']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")
            
    # Project weights (ensure sum(lambda) == 1.0 using normalization if they deviate slightly)
    weights = np.array([lambda1, lambda2, lambda3], dtype=np.float32)
    sum_w = np.sum(weights)
    if not np.isclose(sum_w, 1.0) and sum_w > 0:
        weights = weights / sum_w
        
    l1, l2, l3 = weights
    
    # Compute fusion score: S_fusion = l1 * S_exec + l2 * S_struct + l3 * S_sem
    df['fusion_score'] = (
        l1 * df['exec_norm'].fillna(0.0) +
        l2 * df['struct_norm'].fillna(0.0) +
        l3 * df['semantic_norm'].fillna(0.0)
    )
    
    # Select columns and sort descending by fusion_score
    result = df[['statement_id', 'fusion_score']].copy()
    result = result.sort_values(by='fusion_score', ascending=False).reset_index(drop=True)
    
    return result
