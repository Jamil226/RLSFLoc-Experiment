import os
import sys
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_aeeem_metrics
from src.environment.fl_env import FaultLocalizationEnv
from src.agent.dqn import DQNAgent

def main():
    print("==================================================")
    print("      RLSFLoc AGENT TRAINING SESSION RUNNING      ")
    print("==================================================")
    
    aeeem_file = "./data/processed/aeeem_cleaned.csv"
    if not os.path.exists(aeeem_file):
        print(f"Error: Processed dataset not found at {aeeem_file}. Running with mock data.")
        # Create small mock dataset for validation runs if file missing
        import pandas as pd
        mock_data = pd.DataFrame(np.random.rand(100, 10), columns=[f"feat_{i}" for i in range(10)])
        mock_data['id'] = range(100)
        mock_data['class'] = ['buggy' if i % 5 == 0 else 'clean' for i in range(100)]
        mock_data.to_csv("mock_temp.csv", index=False)
        features, target = load_aeeem_metrics("mock_temp.csv")
        os.remove("mock_temp.csv")
    else:
        features, target = load_aeeem_metrics(aeeem_file)
        
    print(f"Dataset metrics loaded successfully.")
    print(f"Number of classes: {len(features)}")
    print(f"Observation dimension: {features.shape[1]}")
    
    # Initialize env & agent
    env = FaultLocalizationEnv(features, target)
    agent = DQNAgent(state_dim=features.shape[1])
    
    epochs = 10
    batch_size = 32
    
    for epoch in range(epochs):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        losses = []
        
        while not done:
            action = agent.select_action(state, epsilon=0.3)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step(batch_size=batch_size)
            if loss > 0:
                losses.append(loss)
                
            state = next_state
            total_reward += reward
            steps += 1
            
        avg_loss = np.mean(losses) if losses else 0.0
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Total Steps: {steps:3d} | Avg Loss: {avg_loss:6.4f} | Total Reward: {total_reward:6.2f}")
        
    print("\nTraining completed successfully! Saving agent checkpoints...")
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(agent.q_net.state_dict(), os.path.join(checkpoint_dir, "dqn_fl_model.pth"))
    print(f"Checkpoint saved to {checkpoint_dir}/dqn_fl_model.pth")
    print("==================================================")

if __name__ == "__main__":
    main()
