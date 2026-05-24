import numpy as np
import pandas as pd
import time
import tracemalloc

def compute_mrr(rankings):
    """
    Mean Reciprocal Rank (MRR).
    rankings: List of lists containing binary indicators (1 for buggy component, 0 otherwise)
              sorted by candidate ranking.
    """
    rr_list = []
    for rank in rankings:
        indices = np.where(np.array(rank) == 1)[0]
        if len(indices) > 0:
            rr_list.append(1.0 / (indices[0] + 1))
        else:
            rr_list.append(0.0)
    return np.mean(rr_list)

def compute_map(rankings):
    """
    Mean Average Precision (MAP).
    """
    ap_list = []
    for rank in rankings:
        indices = np.where(np.array(rank) == 1)[0]
        if len(indices) == 0:
            ap_list.append(0.0)
            continue
        precisions = [len(np.where(np.array(rank[:idx+1]) == 1)[0]) / (idx + 1) for idx in indices]
        ap_list.append(np.mean(precisions))
    return np.mean(ap_list)

def compute_top_k(rankings, k=1):
    """
    Top-K Accuracy.
    """
    hits = 0
    for rank in rankings:
        if np.sum(rank[:k]) > 0:
            hits += 1
    return hits / len(rankings)

def evaluate_rlsfloc_performance(ranked_dfs, ground_truths, eval_func=None, custom_runtime=None, custom_peak_memory=None):
    """
    Evaluate RLSFLoc localization effectiveness on a suite of bugs.
    
    Parameters:
    -----------
    ranked_dfs : list of pandas.DataFrame
        List of DataFrames representing the fused statement lists for each bug, 
        sorted descending by fusion score. Each DataFrame must contain the column 'statement_id'.
    ground_truths : list of (list or set)
        List of actual ground-truth faulty statement IDs for each corresponding bug.
    eval_func : callable, optional
        A zero-argument function that runs the scoring/ranking process. If provided,
        this function dynamically benchmarks its execution runtime and peak memory usage.
    custom_runtime : float, optional
        Pre-calculated runtime to include in metrics output.
    custom_peak_memory : float, optional
        Pre-calculated peak memory (in MB) to include in metrics output.
        
    Returns:
    --------
    dict
        A dictionary containing all calculated metrics:
        - top_1, top_3, top_5, top_10
        - mrr, map
        - exam_score
        - runtime_sec (optional)
        - peak_memory_mb (optional)
    """
    # 1. Benchmark runtime and memory if eval_func is provided
    benchmarked_lists = None
    runtime_sec = custom_runtime
    peak_memory_mb = custom_peak_memory
    
    if eval_func is not None:
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # Execute the scoring/ranking logic
        benchmarked_lists = eval_func()
        
        end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        runtime_sec = end_time - start_time
        peak_memory_mb = peak / (1024.0 * 1024.0)
        
        # Overwrite ranked_dfs if returned by eval_func
        if benchmarked_lists is not None:
            ranked_dfs = benchmarked_lists
            
    if not ranked_dfs or not ground_truths:
        raise ValueError("ranked_dfs and ground_truths lists cannot be empty")
        
    if len(ranked_dfs) != len(ground_truths):
        raise ValueError("Dimension mismatch: ranked_dfs and ground_truths must have the same length")
        
    mrr_values = []
    ap_values = []
    exam_values = []
    
    top_1_hits = 0
    top_3_hits = 0
    top_5_hits = 0
    top_10_hits = 0
    
    # 2. Iterate through each bug to calculate ranking metrics
    for df, gt in zip(ranked_dfs, ground_truths):
        gt_set = set(gt)
        num_statements = len(df)
        
        if num_statements == 0 or not gt_set:
            continue
            
        statement_list = list(df['statement_id'].values)
        
        # Find 1-based ranks of all ground-truth faults in the ranked list
        fault_ranks = []
        for fault in gt_set:
            if fault in statement_list:
                # 1-based rank index
                rank = statement_list.index(fault) + 1
                fault_ranks.append(rank)
            else:
                # Max rank penalty if fault unmapped
                fault_ranks.append(num_statements + 1)
                
        # First matching fault rank (for Top-k, MRR, EXAM)
        min_rank = min(fault_ranks)
        
        # Top-K hits
        if min_rank <= 1:
            top_1_hits += 1
        if min_rank <= 3:
            top_3_hits += 1
        if min_rank <= 5:
            top_5_hits += 1
        if min_rank <= 10:
            top_10_hits += 1
            
        # MRR: reciprocal of the first matched fault rank
        mrr_values.append(1.0 / min_rank)
        
        # EXAM: inspect effort ratio to find the first fault
        exam_values.append(min_rank / num_statements)
        
        # Average Precision (AP) for this bug
        sorted_ranks = sorted(fault_ranks)
        precisions = []
        for idx, rank in enumerate(sorted_ranks):
            # Number of faults ranked <= current rank is (idx + 1)
            num_faults_above = idx + 1
            precisions.append(num_faults_above / rank)
        ap_values.append(np.mean(precisions) if precisions else 0.0)
        
    num_bugs = len(ranked_dfs)
    
    metrics = {
        "top_1": top_1_hits / num_bugs,
        "top_3": top_3_hits / num_bugs,
        "top_5": top_5_hits / num_bugs,
        "top_10": top_10_hits / num_bugs,
        "mrr": float(np.mean(mrr_values)) if mrr_values else 0.0,
        "map": float(np.mean(ap_values)) if ap_values else 0.0,
        "exam_score": float(np.mean(exam_values)) if exam_values else 0.0
    }
    
    if runtime_sec is not None:
        metrics["runtime_sec"] = float(runtime_sec)
    if peak_memory_mb is not None:
        metrics["peak_memory_mb"] = float(peak_memory_mb)
        
    return metrics
