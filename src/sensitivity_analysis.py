import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.rl_env_gymnasium import RLSFLocEnv
from src.agent.ppo import PPOAgent
from src.utils.metrics import evaluate_rlsfloc_performance
from src.utils.fusion import fuse_suspiciousness_scores

def generate_sensitivity_bug_dataset(num_bugs=40, alpha=0.5, seed=42):
    """
    Generate synthetic bug dataset where the 'struct_norm' column is dynamically
    computed based on the given graph propagation coefficient 'alpha':
    S_struct = (1 - alpha) * S_exec + alpha * S_neighbor
    """
    np.random.seed(seed)
    scores_list = []
    ground_truth_list = []
    
    for bug_idx in range(num_bugs):
        num_statements = np.random.randint(30, 80)
        statement_ids = [f"s{i}" for i in range(num_statements)]
        
        # Pick 1 faulty statement
        faulty_idx = np.random.randint(0, num_statements)
        faulty_id = statement_ids[faulty_idx]
        
        # Ochiai score (exec_norm)
        exec_scores = np.random.beta(2, 5, size=num_statements)
        
        # Simulated neighbor scores representing structural dependency links
        neighbor_scores = np.random.beta(2, 5, size=num_statements)
        
        # Elevate buggy statement scores
        bug_type = bug_idx % 4
        if bug_type == 0:
            exec_scores[faulty_idx] = np.random.uniform(0.85, 1.0)
            neighbor_scores[faulty_idx] = np.random.uniform(0.6, 0.8)
        elif bug_type == 1:
            exec_scores[faulty_idx] = np.random.uniform(0.3, 0.6)
            neighbor_scores[faulty_idx] = np.random.uniform(0.85, 1.0)
        elif bug_type == 2:
            exec_scores[faulty_idx] = np.random.uniform(0.3, 0.6)
            neighbor_scores[faulty_idx] = np.random.uniform(0.3, 0.6)
        else:
            exec_scores[faulty_idx] = np.random.uniform(0.7, 0.9)
            neighbor_scores[faulty_idx] = np.random.uniform(0.7, 0.9)
            
        # Challenge the baseline
        highest_exec_clean_idx = (faulty_idx + 5) % num_statements
        exec_scores[highest_exec_clean_idx] = 0.95
        
        # Compute structural score dynamically based on alpha (Equation 430)
        struct_scores = (1.0 - alpha) * exec_scores + alpha * neighbor_scores
        
        # Semantic score
        sem_scores = np.random.beta(2, 5, size=num_statements)
        if bug_type == 2:
            sem_scores[faulty_idx] = np.random.uniform(0.85, 1.0)
        elif bug_type == 3:
            sem_scores[faulty_idx] = np.random.uniform(0.6, 0.8)
            
        # Min-max normalize scores independently (Equation 459)
        def min_max(vals):
            m, M = np.min(vals), np.max(vals)
            return (vals - m) / (M - m + 1e-12)
            
        df = pd.DataFrame({
            'statement_id': statement_ids,
            'exec_norm': min_max(exec_scores),
            'struct_norm': min_max(struct_scores),
            'semantic_norm': min_max(sem_scores)
        })
        
        scores_list.append(df)
        ground_truth_list.append([faulty_id])
        
    return scores_list, ground_truth_list

def run_grid_search():
    print("==================================================")
    print("      RLSFLoc SENSITIVITY SWEEP INITIALIZED       ")
    print("==================================================")
    
    # Sweep Parameters
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    learning_rates = [0.0005, 0.002, 0.01]
    discount_factors = [0.90, 0.95, 0.99]
    episodes_options = [10, 20, 40]
    
    results = []
    best_reward = -np.inf
    best_config = {}
    
    # Fix random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. First, sweep alpha with a set of default RL hyperparameters to isolate structural behavior
    print("\n[Phase 1] Sweeping Structural Alpha Parameter...")
    default_lr = 0.002
    default_gamma = 0.99
    default_episodes = 20
    
    alpha_mrr_trends = []
    
    for alpha in alphas:
        # Generate dataset using current alpha
        train_scores, train_gt = generate_sensitivity_bug_dataset(num_bugs=30, alpha=alpha, seed=42)
        val_scores, val_gt = generate_sensitivity_bug_dataset(num_bugs=10, alpha=alpha, seed=100)
        
        # Instantiate environment
        train_env = RLSFLocEnv(train_scores, train_gt, k=5)
        val_env = RLSFLocEnv(val_scores, val_gt, k=5)
        
        # Train PPO Agent
        agent = PPOAgent(state_dim=9, action_dim=3, lr=default_lr, gamma=default_gamma)
        
        for epoch in range(default_episodes):
            states, actions, log_probs, rewards = [], [], [], []
            for bug_idx in range(len(train_scores)):
                state, _ = train_env.reset(options={"index": bug_idx})
                action, log_prob, _ = agent.select_action(state)
                _, reward, _, _, _ = train_env.step(action)
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
            agent.update(states, actions, log_probs, rewards)
            
        # Evaluate
        val_rewards = []
        for bug_idx in range(len(val_scores)):
            state, _ = val_env.reset(options={"index": bug_idx})
            with torch.no_grad():
                action = agent.actor(torch.FloatTensor(state)).numpy()
            _, reward, _, _, _ = val_env.step(action)
            val_rewards.append(reward)
            
        mean_reward = float(np.mean(val_rewards))
        alpha_mrr_trends.append(mean_reward)
        print(f"  Alpha: {alpha:.1f} | Validation Avg Reward: {mean_reward:6.3f}")
        
    # 2. Second, run a dense grid search over hyperparameter combinations (fixing optimal alpha or sweeping selectively)
    # To keep execution under 10 seconds, we search combinations selectively on a subset
    print("\n[Phase 2] Running Dense RL Hyperparameter Grid Search...")
    grid_count = 0
    total_grids = len(alphas) * len(learning_rates) * len(discount_factors) * len(episodes_options)
    
    # We will sample/sweep a representative grid to run instantly (~20 combinations)
    # Selected dense sweep
    selected_alphas = [0.2, 0.6, 0.8]
    selected_lrs = [0.0005, 0.002, 0.01]
    selected_gammas = [0.90, 0.99]
    selected_episodes = [10, 20, 40]
    
    for alpha in selected_alphas:
        train_scores, train_gt = generate_sensitivity_bug_dataset(num_bugs=25, alpha=alpha, seed=42)
        val_scores, val_gt = generate_sensitivity_bug_dataset(num_bugs=10, alpha=alpha, seed=100)
        
        train_env = RLSFLocEnv(train_scores, train_gt, k=5)
        val_env = RLSFLocEnv(val_scores, val_gt, k=5)
        
        for lr in selected_lrs:
            for gamma in selected_gammas:
                for eps in selected_episodes:
                    agent = PPOAgent(state_dim=9, action_dim=3, lr=lr, gamma=gamma)
                    
                    # Train
                    for epoch in range(eps):
                        states, actions, log_probs, rewards = [], [], [], []
                        for bug_idx in range(len(train_scores)):
                            state, _ = train_env.reset(options={"index": bug_idx})
                            action, log_prob, _ = agent.select_action(state)
                            _, reward, _, _, _ = train_env.step(action)
                            states.append(state)
                            actions.append(action)
                            log_probs.append(log_prob)
                            rewards.append(reward)
                        agent.update(states, actions, log_probs, rewards)
                        
                    # Evaluate
                    val_rewards = []
                    for bug_idx in range(len(val_scores)):
                        state, _ = val_env.reset(options={"index": bug_idx})
                        with torch.no_grad():
                            action = agent.actor(torch.FloatTensor(state)).numpy()
                        _, reward, _, _, _ = val_env.step(action)
                        val_rewards.append(reward)
                        
                    mean_reward = float(np.mean(val_rewards))
                    
                    results.append({
                        "alpha": alpha,
                        "lr": lr,
                        "gamma": gamma,
                        "episodes": eps,
                        "reward": mean_reward
                    })
                    
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        best_config = {
                            "alpha": alpha,
                            "lr": lr,
                            "gamma": gamma,
                            "episodes": eps,
                            "reward": mean_reward
                        }
                        
    # Sort results by reward
    results_df = pd.DataFrame(results).sort_values(by="reward", ascending=False).reset_index(drop=True)
    
    print("\n==================== DENSE SENSITIVITY GRID SEARCH RESULTS ====================")
    print(results_df.head(10).to_markdown(index=False))
    print("=================================================================================")
    
    print(f"\n==================== BEST HYPERPARAMETER CONFIGURATION ====================")
    print(f"  Alpha (Graph Prop Coefficient):  {best_config['alpha']:.2f}")
    print(f"  Learning Rate (PPO Actor):       {best_config['lr']:.4f}")
    print(f"  Discount Factor (Gamma):         {best_config['gamma']:.2f}")
    print(f"  Training Epochs / Episodes:      {best_config['episodes']:d}")
    print(f"  Best Validation Reward achieved: {best_config['reward']:.4f}")
    print("=============================================================================")
    
    # Save CSV
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, "sensitivity_analysis.csv"), index=False)
    
    # 3. Generate Significance Plots
    print("\nGenerating sensitivity trend plots in results/ directory...")
    
    # A. Alpha Sensitivity Trend Line Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, alpha_mrr_trends, marker='o', linewidth=2.0, color='#3498DB', label='Validation Reward')
    ax.set_xlabel("Graph Propagation Coefficient (alpha)", fontsize=11, fontweight='bold', color='#2C3E50')
    ax.set_ylabel("Average Validation Reward", fontsize=11, fontweight='bold', color='#2C3E50')
    ax.set_title("Sensitivity Sweep: Graph Propagation Strength (alpha)\n(Hyperparameters: LR=0.002, Gamma=0.99, Epochs=20)", fontsize=12, fontweight='bold', pad=12, color='#2C3E50')
    ax.grid(True, linestyle='--', alpha=0.5, color='#BDC3C7')
    ax.set_xticks(alphas)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "alpha_sensitivity_plot.png"), dpi=300)
    plt.close()
    
    # B. Hyperparameter boxplot comparing learning rate performances
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    lr_groups = [results_df[results_df['lr'] == lr]['reward'].values for lr in learning_rates]
    ax2.boxplot(lr_groups, patch_artist=True, tick_labels=[str(lr) for lr in learning_rates],
                medianprops=dict(color='#E74C3C', linewidth=1.5),
                boxprops=dict(facecolor='#AED581', edgecolor='#558B2F', linewidth=1.2))
    ax2.set_xlabel("Learning Rate", fontsize=11, fontweight='bold', color='#2C3E50')
    ax2.set_ylabel("Validation Reward", fontsize=11, fontweight='bold', color='#2C3E50')
    ax2.set_title("Sensitivity Sweep: PPO Learning Rate Distribution", fontsize=12, fontweight='bold', pad=12, color='#2C3E50')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#BDC3C7')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "lr_sensitivity_plot.png"), dpi=300)
    plt.close()
    
    print("Saved plots to results/alpha_sensitivity_plot.png and lr_sensitivity_plot.png successfully.")

if __name__ == "__main__":
    run_grid_search()
