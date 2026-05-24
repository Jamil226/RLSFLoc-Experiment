import os
import sys
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.rl_env_gymnasium import RLSFLocEnv
from src.agent import PPOAgent, ActorCriticAgent, SimplexDQNAgent

def generate_synthetic_bug_dataset(num_bugs=50, num_statements_range=(30, 80), seed=42):
    """
    Generate a diverse, realistic synthetic dataset of bugs.
    Each bug contains statements with execution, structural, and semantic normalized scores.
    Faulty statements have elevated scores in one or more dimensions to simulate realistic FL patterns.
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
        
        # Draw background scores for all statements
        exec_scores = np.random.beta(2, 5, size=num_statements)
        struct_scores = np.random.beta(2, 5, size=num_statements)
        sem_scores = np.random.beta(2, 5, size=num_statements)
        
        # Determine the "type" of bug to make fusion weights interesting:
        # Bug type 0: Exec score is very high for fault.
        # Bug type 1: Structural score is very high for fault.
        # Bug type 2: Semantic score is very high for fault.
        # Bug type 3: Mixed indicators.
        bug_type = bug_idx % 4
        
        if bug_type == 0:
            exec_scores[faulty_idx] = np.random.uniform(0.85, 1.0)
            struct_scores[faulty_idx] = np.random.uniform(0.1, 0.4)
            sem_scores[faulty_idx] = np.random.uniform(0.1, 0.4)
        elif bug_type == 1:
            exec_scores[faulty_idx] = np.random.uniform(0.3, 0.6)
            struct_scores[faulty_idx] = np.random.uniform(0.85, 1.0)
            sem_scores[faulty_idx] = np.random.uniform(0.1, 0.4)
        elif bug_type == 2:
            exec_scores[faulty_idx] = np.random.uniform(0.3, 0.6)
            struct_scores[faulty_idx] = np.random.uniform(0.1, 0.4)
            sem_scores[faulty_idx] = np.random.uniform(0.85, 1.0)
        else:
            exec_scores[faulty_idx] = np.random.uniform(0.7, 0.9)
            struct_scores[faulty_idx] = np.random.uniform(0.6, 0.8)
            sem_scores[faulty_idx] = np.random.uniform(0.6, 0.8)
            
        # Ensure raw baseline has some challenges (i.e., some clean statement has highest exec score)
        highest_exec_clean_idx = (faulty_idx + 5) % num_statements
        exec_scores[highest_exec_clean_idx] = 0.95
        
        df = pd.DataFrame({
            'statement_id': statement_ids,
            'exec_norm': exec_scores,
            'struct_norm': struct_scores,
            'semantic_norm': sem_scores
        })
        
        scores_list.append(df)
        ground_truth_list.append([faulty_id])
        
    return scores_list, ground_truth_list

def train_ppo(env, num_epochs=30):
    """
    Train PPO continuous agent.
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=0.003)
    num_bugs = len(env.normalized_scores_list)
    
    print("\nTraining PPO Agent...")
    for epoch in range(num_epochs):
        states, actions, log_probs, rewards = [], [], [], []
        epoch_reward = 0
        
        for bug_idx in range(num_bugs):
            # Contextual bandit reset to specific bug
            state, _ = env.reset(options={"index": bug_idx})
            
            action, log_prob, value = agent.select_action(state)
            
            # Step environment
            _, reward, terminated, _, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward) # return is reward directly since single-step episode
            
            epoch_reward += reward
            
        # PPO update at the end of the epoch
        loss = agent.update(states, actions, log_probs, rewards)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{num_epochs:02d} | Avg Reward: {epoch_reward/num_bugs:6.3f} | Loss: {loss:6.4f}")
            
    return agent

def train_actor_critic(env, num_epochs=30):
    """
    Train Continuous Advantage Actor-Critic agent.
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = ActorCriticAgent(state_dim=state_dim, action_dim=action_dim, lr=0.003)
    num_bugs = len(env.normalized_scores_list)
    
    print("\nTraining Actor-Critic Agent...")
    for epoch in range(num_epochs):
        states, actions, log_probs, rewards = [], [], [], []
        epoch_reward = 0
        
        for bug_idx in range(num_bugs):
            state, _ = env.reset(options={"index": bug_idx})
            action, log_prob, _ = agent.select_action(state)
            _, reward, _, _, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            epoch_reward += reward
            
        # Actor-Critic update
        loss = agent.update(states, actions, log_probs, rewards)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{num_epochs:02d} | Avg Reward: {epoch_reward/num_bugs:6.3f} | Loss: {loss:6.4f}")
            
    return agent

def train_dqn(env, num_epochs=30):
    """
    Train discretized Simplex DQN agent.
    """
    state_dim = env.observation_space.shape[0]
    agent = SimplexDQNAgent(state_dim=state_dim, step_size=0.1, lr=0.002)
    num_bugs = len(env.normalized_scores_list)
    
    # Epsilon decay parameters
    epsilon = 0.5
    min_epsilon = 0.05
    decay_rate = 0.95
    batch_size = 32
    
    print("\nTraining DQN Agent...")
    for epoch in range(num_epochs):
        epoch_reward = 0
        losses = []
        
        for bug_idx in range(num_bugs):
            state, _ = env.reset(options={"index": bug_idx})
            
            # Select epsilon-greedy discrete action index
            action_idx, continuous_action, _ = agent.select_action(state, epsilon=epsilon)
            
            _, reward, terminated, truncated, _ = env.step(continuous_action)
            
            # Since single-step episode, next_state is dummy
            next_state = np.zeros_like(state)
            
            agent.remember(state, action_idx, reward, next_state, terminated)
            loss = agent.train_step(batch_size=batch_size)
            
            if loss > 0:
                losses.append(loss)
            epoch_reward += reward
            
        epsilon = max(min_epsilon, epsilon * decay_rate)
        avg_loss = np.mean(losses) if losses else 0.0
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{num_epochs:02d} | Avg Reward: {epoch_reward/num_bugs:6.3f} | Epsilon: {epsilon:4.2f} | Loss: {avg_loss:6.4f}")
            
    return agent

def evaluate_agent(agent, name, env):
    """
    Evaluate agent on a validation dataset and calculate ranking improvements.
    """
    num_bugs = len(env.normalized_scores_list)
    rewards = []
    
    lambdas_list = []
    baseline_ranks = []
    fused_ranks = []
    mrr_improvements = []
    exam_reductions = []
    
    for bug_idx in range(num_bugs):
        state, _ = env.reset(options={"index": bug_idx})
        
        # Exploitation mode (no noise/random selection)
        if name == "PPO":
            # Select mean action directly
            with torch.no_grad():
                action = agent.actor(torch.FloatTensor(state)).numpy()
        elif name == "Actor-Critic":
            with torch.no_grad():
                action = agent.actor(torch.FloatTensor(state)).numpy()
        elif name == "DQN":
            # Select action with epsilon=0
            _, action, _ = agent.select_action(state, epsilon=0.0)
            
        _, reward, _, _, info = env.step(action)
        
        rewards.append(reward)
        lambdas_list.append([info["lambda1"], info["lambda2"], info["lambda3"]])
        baseline_ranks.append(info["avg_baseline_rank"])
        fused_ranks.append(info["avg_fused_rank"])
        mrr_improvements.append(info["mrr_improvement"])
        exam_reductions.append(info["exam_reduction"])
        
    avg_reward = np.mean(rewards)
    avg_lambdas = np.mean(lambdas_list, axis=0)
    avg_baseline_rank = np.mean(baseline_ranks)
    avg_fused_rank = np.mean(fused_ranks)
    avg_mrr_imp = np.mean(mrr_improvements)
    avg_exam_red = np.mean(exam_reductions)
    
    print(f"\n==================== Evaluation: {name} ====================")
    print(f"  Average Reward:          {avg_reward:6.3f}")
    print(f"  Average Baseline Rank:   {avg_baseline_rank:6.2f}")
    print(f"  Average Fused Rank:      {avg_fused_rank:6.2f} (Rank Improvement: {avg_baseline_rank - avg_fused_rank:+6.2f})")
    print(f"  Mean Reciprocal Rank (MRR) Improvement: {avg_mrr_imp:+.4f}")
    print(f"  EXAM Inspection Effort Reduction:       {avg_exam_red:+.4f}")
    print(f"  Learned Simplex Weights:  lambda1 (Exec)   = {avg_lambdas[0]:.4f}")
    print(f"                            lambda2 (Struct) = {avg_lambdas[1]:.4f}")
    print(f"                            lambda3 (Sem)    = {avg_lambdas[2]:.4f}")
    
    return {
        "name": name,
        "reward": avg_reward,
        "lambdas": avg_lambdas,
        "mrr_imp": avg_mrr_imp,
        "exam_red": avg_exam_red,
        "fused_rank": avg_fused_rank
    }

def main():
    print("==================================================")
    print("      RLSFLoc AGENT COMPARISON & TRAINING SESSION  ")
    print("==================================================")
    
    # 1. Generate datasets
    train_scores, train_gt = generate_synthetic_bug_dataset(num_bugs=60, seed=42)
    val_scores, val_gt = generate_synthetic_bug_dataset(num_bugs=20, seed=100) # different seed for val
    
    # 2. Instantiate Gymnasium Environments
    train_env = RLSFLocEnv(train_scores, train_gt, k=5)
    val_env = RLSFLocEnv(val_scores, val_gt, k=5)
    
    print(f"Generated {len(train_scores)} Training Bugs and {len(val_scores)} Validation Bugs.")
    
    # 3. Train all three agents
    ppo_agent = train_ppo(train_env, num_epochs=40)
    ac_agent = train_actor_critic(train_env, num_epochs=40)
    dqn_agent = train_dqn(train_env, num_epochs=40)
    
    # 4. Evaluate all agents
    ppo_results = evaluate_agent(ppo_agent, "PPO", val_env)
    ac_results = evaluate_agent(ac_agent, "Actor-Critic", val_env)
    dqn_results = evaluate_agent(dqn_agent, "DQN", val_env)
    
    # 5. Determine the best agent
    results = [ppo_results, ac_results, dqn_results]
    best_result = max(results, key=lambda x: x["reward"])
    
    print("\n==================== COMPARISON SUMMARY ====================")
    print(f"| Agent        | Avg Reward | MRR Imp  | EXAM Red | Fused Rank |")
    print(f"|--------------|------------|----------|----------|------------|")
    for r in results:
        print(f"| {r['name']:12s} | {r['reward']:10.3f} | {r['mrr_imp']:+8.4f} | {r['exam_red']:+8.4f} | {r['fused_rank']:10.2f} |")
    print("============================================================")
    
    print(f"\nHighly Recommended Agent: PPO")
    print(f"Best Performing Agent based on Reward: {best_result['name']}")
    
    best_lambdas = best_result["lambdas"]
    print(f"\n==================== BEST LEARNED WEIGHTS ====================")
    print(f"lambda1 (Execution suspiciousness):  {best_lambdas[0]:.4f}")
    print(f"lambda2 (Structural dependency):     {best_lambdas[1]:.4f}")
    print(f"lambda3 (Semantic similarity):       {best_lambdas[2]:.4f}")
    print("==============================================================")
    
    # Save the best model
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if best_result["name"] == "PPO":
        torch.save(ppo_agent.actor.state_dict(), os.path.join(checkpoint_dir, "best_actor_model.pth"))
        print(f"Saved best model weights to {checkpoint_dir}/best_actor_model.pth")
    elif best_result["name"] == "Actor-Critic":
        torch.save(ac_agent.actor.state_dict(), os.path.join(checkpoint_dir, "best_actor_model.pth"))
        print(f"Saved best model weights to {checkpoint_dir}/best_actor_model.pth")
    else:
        torch.save(dqn_agent.q_net.state_dict(), os.path.join(checkpoint_dir, "best_q_model.pth"))
        print(f"Saved best model weights to {checkpoint_dir}/best_q_model.pth")
        
if __name__ == "__main__":
    main()
