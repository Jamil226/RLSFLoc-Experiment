import numpy as np
import pandas as pd

def normalize_score_input(scores, col_name="score"):
    """
    Standardizes various pandas/dict inputs into a clean dictionary mapping statement_id -> score.
    """
    if isinstance(scores, pd.DataFrame):
        cols = list(scores.columns)
        if 'statement_id' in cols:
            stmt_col = 'statement_id'
            score_col = [c for c in cols if c != stmt_col][0]
        else:
            stmt_col = cols[0]
            score_col = cols[1]
        return dict(zip(scores[stmt_col], scores[score_col]))
    elif isinstance(scores, pd.Series):
        return scores.to_dict()
    elif isinstance(scores, dict):
        return scores
    else:
        raise TypeError(f"scores input for {col_name} must be a dict, pandas Series, or pandas DataFrame")


def normalize_rlsfloc_scores(execution_scores, structural_scores, semantic_scores, epsilon=1e-12):
    """
    Applies min-max normalization independently to Execution, Structural, and Semantic
    suspiciousness scores as defined in Equations 459 and 460 of the RLSFLoc research article:

    S_hat_k(v_i) = (S_k(v_i) - min(S_k)) / (max(S_k) - min(S_k) + epsilon)

    Parameters:
    -----------
    execution_scores : dict, pandas.Series, or pandas.DataFrame
        Raw execution suspiciousness scores (e.g. Ochiai).
    structural_scores : dict, pandas.Series, or pandas.DataFrame
        Raw structural suspiciousness scores (propagated).
    semantic_scores : dict, pandas.Series, or pandas.DataFrame
        Raw semantic similarity scores (SentenceTransformer cosine similarities).
    epsilon : float, optional (default=1e-12)
        A small constant added to denominators to avoid division by zero.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: ['statement_id', 'exec_norm', 'struct_norm', 'semantic_norm']
        Aligned by statement_id and sorted in their original/union ordering.
    """
    # 1. Standardize all three score inputs into dictionaries
    exec_dict = normalize_score_input(execution_scores, "execution_scores")
    struct_dict = normalize_score_input(structural_scores, "structural_scores")
    sem_dict = normalize_score_input(semantic_scores, "semantic_scores")

    # 2. Extract union of all statement IDs to ensure perfect alignment
    all_statement_ids = list(set(exec_dict.keys()) | set(struct_dict.keys()) | set(sem_dict.keys()))
    # Sort statement IDs to guarantee stable ordering
    all_statement_ids.sort()

    if not all_statement_ids:
        return pd.DataFrame(columns=['statement_id', 'exec_norm', 'struct_norm', 'semantic_norm'])

    # 3. Pull raw scores into aligned numpy arrays
    raw_exec = np.array([float(exec_dict.get(sid, 0.0)) for sid in all_statement_ids])
    raw_struct = np.array([float(struct_dict.get(sid, 0.0)) for sid in all_statement_ids])
    raw_sem = np.array([float(sem_dict.get(sid, 0.0)) for sid in all_statement_ids])

    # 4. Perform min-max normalization independently per array (Equation 459)
    def min_max(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        denom = max_val - min_val
        return (arr - min_val) / (denom + epsilon)

    norm_exec = min_max(raw_exec)
    norm_struct = min_max(raw_struct)
    norm_sem = min_max(raw_sem)

    # 5. Package results in a clean DataFrame matching exact output format
    norm_df = pd.DataFrame({
        'statement_id': all_statement_ids,
        'exec_norm': norm_exec,
        'struct_norm': norm_struct,
        'semantic_norm': norm_sem
    })

    return norm_df
