import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_aeeem_metrics
from src.environment.fl_env import FaultLocalizationEnv
from src.agent.dqn import DQNAgent
from src.utils.metrics import compute_mrr, compute_map, compute_top_k

def main():
    print("==================================================")
    print("      RLSFLoc AGENT EVALUATION RUNNING            ")
    print("==================================================")
    
    aeeem_file = "./data/processed/aeeem_cleaned.csv"
    if not os.path.exists(aeeem_file):
        print(f"Error: dataset file not found at {aeeem_file}. Exiting.")
        sys.exit(1)
        
    features, target = load_aeeem_metrics(aeeem_file)
    env = FaultLocalizationEnv(features, target)
    agent = DQNAgent(state_dim=features.shape[1])
    
    # Load trained model if exists
    model_path = "./checkpoints/dqn_fl_model.pth"
    if os.path.exists(model_path):
        print(f"Loading trained agent weights from {model_path}...")
        agent.q_net.load_state_dict(torch.load(model_path))
    else:
        print("Warning: Trained weights not found. Evaluating with random/untrained weights.")
        
    # Evaluate over AEEEM Metrics
    state = env.reset()
    done = False
    
    rankings = []
    
    while not done:
        # Exploit (no exploration, epsilon=0)
        action = agent.select_action(state, epsilon=0.0)
        label = target.iloc[env.current_idx]
        
        # Record ranking predictions (action represents top-rank decision)
        rankings.append(1 if (action == 1 and label == 1) else 0)
        
        next_state, reward, done, _ = env.step(action)
        state = next_state
        
    # Standard evaluation ranking list format (batch size is 1 episode)
    eval_rankings = [rankings]
    
    # Calculate Metrics
    mrr = compute_mrr(eval_rankings)
    m_ap = compute_map(eval_rankings)
    top_1 = compute_top_k(eval_rankings, k=1)
    top_5 = compute_top_k(eval_rankings, k=5)
    
    print("\nEvaluation Results:")
    print(f"  - Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"  - Mean Average Precision (MAP): {m_ap:.4f}")
    print(f"  - Top-1 Accuracy: {top_1 * 100:.2f}%")
    print(f"  - Top-5 Accuracy: {top_5 * 100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
