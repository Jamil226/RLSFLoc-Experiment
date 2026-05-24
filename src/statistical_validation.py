import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate_baselines import generate_comparative_bug_dataset
from src.agent.ppo import PPOAgent
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

def cliffs_delta(x, y):
    """
    Computes Cliff's Delta effect size between two non-parametric distributions.
    d = [sum_{i,j} sign(x_i - y_j)] / (m * n)
    """
    m, n = len(x), len(y)
    diff = 0
    for val_x in x:
        for val_y in y:
            if val_x > val_y:
                diff += 1
            elif val_x < val_y:
                diff -= 1
    d = diff / (m * n)
    
    # Standard literature effect size thresholds (Cliff, 1993)
    abs_d = abs(d)
    if abs_d < 0.147:
        effect = "Negligible"
    elif abs_d < 0.33:
        effect = "Small"
    elif abs_d < 0.474:
        effect = "Medium"
    else:
        effect = "Large"
        
    return d, effect

def main():
    print("==================================================")
    print("      RLSFLoc STATISTICAL VALIDATION ENGINE       ")
    print("==================================================")
    
    # 1. Generate datasets
    train_scores, train_gt = generate_comparative_bug_dataset(num_bugs=60, seed=42)
    val_scores, val_gt = generate_comparative_bug_dataset(num_bugs=20, seed=100)
    
    # 2. Fit deep learning baselines
    print("Fitting baseline models on training data...")
    mlp_baseline = DeepFLMLPBaseline(lr=0.005, epochs=25)
    mlp_baseline.fit(train_scores, train_gt)
    
    ranknet_baseline = RankNetBaseline(lr=0.005, epochs=25, num_pairs_per_bug=50)
    ranknet_baseline.fit(train_scores, train_gt)
    
    # 3. Load PPO agent
    ppo_agent = PPOAgent(state_dim=9, action_dim=3)
    ppo_actor_path = "./checkpoints/best_actor_model.pth"
    if os.path.exists(ppo_actor_path):
        ppo_agent.actor.load_state_dict(torch.load(ppo_actor_path))
    else:
        # Mini train sequence if checkpoint missing
        from src.environment.rl_env_gymnasium import RLSFLocEnv
        train_env = RLSFLocEnv(train_scores, train_gt, k=5)
        for epoch in range(20):
            states, actions, log_probs, rewards = [], [], [], []
            for bug_idx in range(len(train_scores)):
                state, _ = train_env.reset(options={"index": bug_idx})
                action, log_prob, _ = ppo_agent.select_action(state)
                _, reward, _, _, _ = train_env.step(action)
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
            ppo_agent.update(states, actions, log_probs, rewards)
            
    # 4. Extract statement-by-statement ranks for all methods
    methods = {
        "Ochiai": lambda df: get_ochiai_baseline(df),
        "Tarantula": lambda df: get_tarantula_baseline(df),
        "DStar": lambda df: get_dstar_baseline(df),
        "Graph-Only": lambda df: get_graph_baseline(df),
        "Transformer-Only": lambda df: get_transformer_baseline(df),
        "DeepFL-like MLP": lambda df: mlp_baseline.predict(df),
        "RankNet LTR": lambda df: ranknet_baseline.predict(df),
    }
    
    # Add RLSFLoc
    def rlsfloc_predict(df):
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
        
        with torch.no_grad():
            action = ppo_agent.actor(torch.FloatTensor(state)).numpy()
        exp_act = np.exp(action - np.max(action))
        lambdas = exp_act / (np.sum(exp_act) + 1e-12)
        return fuse_suspiciousness_scores(df, lambda1=lambdas[0], lambda2=lambdas[1], lambda3=lambdas[2])
        
    methods["RLSFLoc (PPO)"] = rlsfloc_predict
    
    # Record Reciprocal Ranks and EXAM scores per bug per method
    rr_data = {name: [] for name in methods.keys()}
    exam_data = {name: [] for name in methods.keys()}
    
    print("\nRunning inference and calculating ranks...")
    for df, gt in zip(val_scores, val_gt):
        gt_set = set(gt)
        num_statements = len(df)
        
        for name, predict_func in methods.items():
            fused = predict_func(df)
            statement_list = list(fused['statement_id'].values)
            
            # Find 1-based ranks of faults
            ranks = []
            for fault in gt_set:
                if fault in statement_list:
                    ranks.append(statement_list.index(fault) + 1)
                else:
                    ranks.append(num_statements + 1)
                    
            min_rank = min(ranks) if ranks else num_statements + 1
            
            # Reciprocal Rank
            rr_data[name].append(1.0 / min_rank)
            # EXAM score
            exam_data[name].append(min_rank / num_statements)
            
    # 5. Statistical Calculations
    rlsfloc_name = "RLSFLoc (PPO)"
    rlsfloc_rr = np.array(rr_data[rlsfloc_name])
    rlsfloc_exam = np.array(exam_data[rlsfloc_name])
    
    stats_rows = []
    
    for name in methods.keys():
        rr_vals = np.array(rr_data[name])
        exam_vals = np.array(exam_data[name])
        
        mean_rr, std_rr = np.mean(rr_vals), np.std(rr_vals)
        mean_ex, std_ex = np.mean(exam_vals), np.std(exam_vals)
        
        if name == rlsfloc_name:
            p_val = "-"
            delta_val = "-"
            effect = "-"
        else:
            # Paired Wilcoxon test on Reciprocal Ranks
            # If the vectors are exactly equal, wilcoxon raises a ValueError. Handle gracefully.
            if np.allclose(rlsfloc_rr, rr_vals):
                p_val = "1.0000"
            else:
                try:
                    _, p = wilcoxon(rlsfloc_rr, rr_vals)
                    p_val = f"{p:.4f}"
                except ValueError:
                    p_val = "1.0000"
                    
            # Cliff's Delta on Reciprocal Ranks
            delta_val, effect = cliffs_delta(rlsfloc_rr, rr_vals)
            delta_val = f"{delta_val:+.4f}"
            
        stats_rows.append({
            "Method": name,
            "MRR (Mean ± Std)": f"{mean_rr:.4f} ± {std_rr:.4f}",
            "EXAM (Mean ± Std)": f"{mean_ex:.4f} ± {std_ex:.4f}",
            "Wilcoxon p-value": p_val,
            "Cliff's Delta d": delta_val,
            "Effect Size": effect
        })
        
    stats_df = pd.DataFrame(stats_rows)
    print("\n================================ PUBLICATION-QUALITY STATISTICAL REPORT ================================")
    print(stats_df.to_markdown(index=False))
    print("=========================================================================================================")
    
    # Save report to CSV
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(results_dir, "statistical_validation.csv"), index=False)
    
    # 6. Generate Significance Plots
    print("\nGenerating significance plots in results/ directory...")
    
    # Order methods for plotting
    plot_names = ["Ochiai", "Tarantula", "DStar", "Graph-Only", "Transformer-Only", "DeepFL-like MLP", "RankNet LTR", "RLSFLoc (PPO)"]
    plot_data = [rr_data[name] for name in plot_names]
    
    # Matplotlib styling for publication quality
    plt.rcParams['font.sans-serif'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # A. Reciprocal Rank Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    box = ax.boxplot(plot_data, patch_artist=True, tick_labels=plot_names,
                     medianprops=dict(color='#E74C3C', linewidth=1.5),
                     flierprops=dict(marker='o', markersize=6, markerfacecolor='#B0BEC5', markeredgecolor='none'))
                     
    # Set premium muted colors
    colors = ['#CFD8DC', '#CFD8DC', '#CFD8DC', '#FFE082', '#FFE082', '#FFCC80', '#90CAF9', '#A5D6A7']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#546E7A')
        patch.set_linewidth(1.2)
        
    ax.set_ylabel("Reciprocal Rank (1 / Rank)", fontsize=12, fontweight='bold', color='#263238')
    ax.set_xlabel("Localizer Model / Baseline", fontsize=12, fontweight='bold', color='#263238')
    ax.set_title("Significance Analysis: Fault Reciprocal Rank Distribution\n(RLSFLoc vs Baselines)", fontsize=14, fontweight='bold', pad=15, color='#263238')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#CFD8DC')
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, fontsize=10, color='#37474F')
    plt.yticks(fontsize=10, color='#37474F')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "significance_plot.png"), dpi=300)
    plt.close()
    
    # B. EXAM Score Boxplot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_data_ex = [exam_data[name] for name in plot_names]
    
    box2 = ax2.boxplot(plot_data_ex, patch_artist=True, tick_labels=plot_names,
                       medianprops=dict(color='#E74C3C', linewidth=1.5),
                       flierprops=dict(marker='o', markersize=6, markerfacecolor='#B0BEC5', markeredgecolor='none'))
                       
    # Re-use premium colors
    for patch, color in zip(box2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#546E7A')
        patch.set_linewidth(1.2)
        
    ax2.set_ylabel("EXAM Score (Rank / N)", fontsize=12, fontweight='bold', color='#263238')
    ax2.set_xlabel("Localizer Model / Baseline", fontsize=12, fontweight='bold', color='#263238')
    ax2.set_title("Significance Analysis: Developer Inspection Effort (EXAM)\n(RLSFLoc vs Baselines)", fontsize=14, fontweight='bold', pad=15, color='#263238')
    
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5, color='#CFD8DC')
    ax2.set_axisbelow(True)
    plt.xticks(rotation=20, fontsize=10, color='#37474F')
    plt.yticks(fontsize=10, color='#37474F')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exam_significance_plot.png"), dpi=300)
    plt.close()
    
    print("Successfully generated and saved plots to results/significance_plot.png and exam_significance_plot.png")
    
if __name__ == "__main__":
    main()
