import os
import sys
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.rl_env_gymnasium import RLSFLocEnv
from src.agent.ppo import PPOAgent
from src.utils.metrics import evaluate_rlsfloc_performance
from src.utils.baselines import (
    DeepFLMLPBaseline,
    RankNetBaseline,
    get_ochiai_baseline,
    get_tarantula_baseline,
    get_dstar_baseline,
    get_graph_baseline,
    get_transformer_baseline
)
from src.utils.fusion import fuse_suspiciousness_scores

def generate_comparative_bug_dataset(num_bugs=50, num_statements_range=(30, 80), seed=42):
    """
    Generate synthetic bug dataset featuring distinct execution formulas
    (Ochiai, Tarantula, DStar) as separate normalized columns to evaluate SBFL properly.
    """
    np.random.seed(seed)
    scores_list = []
    ground_truth_list = []
    
    for bug_idx in range(num_bugs):
        num_statements = np.random.randint(num_statements_range[0], num_statements_range[1] + 1)
        statement_ids = [f"s{i}" for i in range(num_statements)]
        
        # Pick 1 faulty statement
        faulty_idx = np.random.randint(0, num_statements)
        faulty_id = statement_ids[faulty_idx]
        
        # Background Ochiai score (exec_norm)
        exec_scores = np.random.beta(2, 5, size=num_statements)
        
        # Tarantula score: slightly lower resolution under ties (lower rank isolation)
        tarantula_scores = exec_scores.copy()
        tarantula_scores += np.random.normal(0.0, 0.05, size=num_statements)
        tarantula_scores = np.clip(tarantula_scores, 0.0, 1.0)
        
        # DStar score: quadratic confidence amplification
        dstar_scores = exec_scores ** 1.5
        
        struct_scores = np.random.beta(2, 5, size=num_statements)
        sem_scores = np.random.beta(2, 5, size=num_statements)
        
        # Elevate scores for buggy statement
        bug_type = bug_idx % 4
        if bug_type == 0:
            exec_scores[faulty_idx] = np.random.uniform(0.85, 1.0)
            tarantula_scores[faulty_idx] = np.random.uniform(0.75, 0.95)
            dstar_scores[faulty_idx] = np.random.uniform(0.85, 1.0)
            struct_scores[faulty_idx] = np.random.uniform(0.1, 0.4)
            sem_scores[faulty_idx] = np.random.uniform(0.1, 0.4)
        elif bug_type == 1:
            exec_scores[faulty_idx] = np.random.uniform(0.3, 0.6)
            tarantula_scores[faulty_idx] = np.random.uniform(0.2, 0.5)
            dstar_scores[faulty_idx] = np.random.uniform(0.2, 0.6)
            struct_scores[faulty_idx] = np.random.uniform(0.85, 1.0)
            sem_scores[faulty_idx] = np.random.uniform(0.1, 0.4)
        elif bug_type == 2:
            exec_scores[faulty_idx] = np.random.uniform(0.3, 0.6)
            tarantula_scores[faulty_idx] = np.random.uniform(0.2, 0.5)
            dstar_scores[faulty_idx] = np.random.uniform(0.2, 0.6)
            struct_scores[faulty_idx] = np.random.uniform(0.1, 0.4)
            sem_scores[faulty_idx] = np.random.uniform(0.85, 1.0)
        else:
            exec_scores[faulty_idx] = np.random.uniform(0.7, 0.9)
            tarantula_scores[faulty_idx] = np.random.uniform(0.6, 0.8)
            dstar_scores[faulty_idx] = np.random.uniform(0.7, 0.9)
            struct_scores[faulty_idx] = np.random.uniform(0.6, 0.8)
            sem_scores[faulty_idx] = np.random.uniform(0.6, 0.8)
            
        # Challenge the baseline by putting a high background execution score
        highest_exec_clean_idx = (faulty_idx + 5) % num_statements
        exec_scores[highest_exec_clean_idx] = 0.95
        tarantula_scores[highest_exec_clean_idx] = 0.95
        dstar_scores[highest_exec_clean_idx] = 0.95
        
        df = pd.DataFrame({
            'statement_id': statement_ids,
            'exec_norm': exec_scores,
            'tarantula_norm': tarantula_scores,
            'dstar_norm': dstar_scores,
            'struct_norm': struct_scores,
            'semantic_norm': sem_scores
        })
        
        scores_list.append(df)
        ground_truth_list.append([faulty_id])
        
    return scores_list, ground_truth_list

def main():
    print("==================================================")
    print("      RLSFLoc COMPARATIVE BASELINES RUNNING       ")
    print("==================================================")
    
    # 1. Generate datasets
    train_scores, train_gt = generate_comparative_bug_dataset(num_bugs=60, seed=42)
    val_scores, val_gt = generate_comparative_bug_dataset(num_bugs=20, seed=100)
    
    print(f"Generated {len(train_scores)} Training Bugs and {len(val_scores)} Validation Bugs.")
    
    # 2. Train baseline models
    print("\nTraining DeepFL-like MLP Baseline...")
    mlp_baseline = DeepFLMLPBaseline(lr=0.005, epochs=25)
    mlp_baseline.fit(train_scores, train_gt)
    
    print("Training RankNet LTR Baseline...")
    ranknet_baseline = RankNetBaseline(lr=0.005, epochs=25, num_pairs_per_bug=50)
    ranknet_baseline.fit(train_scores, train_gt)
    
    # 3. Load or train PPO Agent
    ppo_actor_path = "./checkpoints/best_actor_model.pth"
    ppo_agent = PPOAgent(state_dim=9, action_dim=3)
    if os.path.exists(ppo_actor_path):
        print("\nLoading pre-trained PPO Agent weights...")
        ppo_agent.actor.load_state_dict(torch.load(ppo_actor_path))
    else:
        print("\nWarning: Pre-trained PPO weights not found. Training quick PPO agent...")
        # Brief train sequence
        train_env = RLSFLocEnv(train_scores, train_gt, k=5)
        num_bugs = len(train_scores)
        for epoch in range(25):
            states, actions, log_probs, rewards = [], [], [], []
            for bug_idx in range(num_bugs):
                state, _ = train_env.reset(options={"index": bug_idx})
                action, log_prob, _ = ppo_agent.select_action(state)
                _, reward, _, _, _ = train_env.step(action)
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
            ppo_agent.update(states, actions, log_probs, rewards)
            
    # 4. Comparative Evaluation
    print("\n==================== Evaluating All Methods ====================")
    
    methods = {
        "Ochiai": lambda: [get_ochiai_baseline(df) for df in val_scores],
        "Tarantula": lambda: [get_tarantula_baseline(df) for df in val_scores],
        "DStar": lambda: [get_dstar_baseline(df) for df in val_scores],
        "Graph-Only": lambda: [get_graph_baseline(df) for df in val_scores],
        "Transformer-Only": lambda: [get_transformer_baseline(df) for df in val_scores],
        "DeepFL-like MLP": lambda: [mlp_baseline.predict(df) for df in val_scores],
        "RankNet LTR": lambda: [ranknet_baseline.predict(df) for df in val_scores],
    }
    
    # Add RLSFLoc (PPO Fusion)
    def rlsfloc_eval():
        ranked_dfs = []
        for df in val_scores:
            # We compute the codebase context state
            # Mean, std, max for each score
            mean_exec = float(df['exec_norm'].mean())
            std_exec = float(df['exec_norm'].std()) if len(df) > 1 else 0.0
            max_exec = float(df['exec_norm'].max())
            mean_struct = float(df['struct_norm'].mean())
            std_struct = float(df['struct_norm'].std()) if len(df) > 1 else 0.0
            max_struct = float(df['struct_norm'].max())
            mean_sem = float(df['semantic_norm'].mean())
            std_sem = float(df['semantic_norm'].std()) if len(df) > 1 else 0.0
            max_sem = float(df['semantic_norm'].max())
            
            state = np.array([
                mean_exec, std_exec, max_exec,
                mean_struct, std_struct, max_struct,
                mean_sem, std_sem, max_sem
            ], dtype=np.float32)
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Predict fusion weights using PPO actor
            with torch.no_grad():
                action = ppo_agent.actor(torch.FloatTensor(state)).numpy()
                
            # Softmax to simplex
            exp_act = np.exp(action - np.max(action))
            lambdas = exp_act / (np.sum(exp_act) + 1e-12)
            
            # Fuse
            ranked_dfs.append(fuse_suspiciousness_scores(df, lambda1=lambdas[0], lambda2=lambdas[1], lambda3=lambdas[2]))
        return ranked_dfs
        
    methods["RLSFLoc (PPO)"] = rlsfloc_eval
    
    report_rows = []
    
    for name, eval_func in methods.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_rlsfloc_performance([], val_gt, eval_func=eval_func)
        report_rows.append({
            "Method": name,
            "Top-1": metrics["top_1"],
            "Top-3": metrics["top_3"],
            "Top-5": metrics["top_5"],
            "Top-10": metrics["top_10"],
            "MRR": metrics["mrr"],
            "MAP": metrics["map"],
            "EXAM": metrics["exam_score"],
            "Runtime (s)": metrics["runtime_sec"],
            "Peak Mem (MB)": metrics["peak_memory_mb"]
        })
        
    report_df = pd.DataFrame(report_rows)
    # Re-order columns
    cols = ["Method", "Top-1", "Top-3", "Top-5", "Top-10", "MRR", "MAP", "EXAM", "Runtime (s)", "Peak Mem (MB)"]
    report_df = report_df[cols]
    
    print("\n================================== COMPARATIVE ANALYSIS REPORT ==================================")
    print(report_df.to_string(index=False))
    print("==================================================================================================")
    
if __name__ == "__main__":
    main()
