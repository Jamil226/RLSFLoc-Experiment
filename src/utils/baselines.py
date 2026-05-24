import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random

# =====================================================================
# 1. DeepFL-like MLP Baseline
# =====================================================================
class MLPNet(nn.Module):
    """
    MLP network architecture similar to DeepFL.
    Predicts fault probability from execution, structural, and semantic normalized scores.
    """
    def __init__(self, input_dim=3):
        super(MLPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class DeepFLMLPBaseline:
    """
    DeepFL-like MLP Classification Baseline.
    """
    def __init__(self, lr=0.005, epochs=15):
        self.epochs = epochs
        self.model = MLPNet(input_dim=3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        
    def fit(self, scores_list, ground_truth_list):
        """
        Train the MLP model using binary cross-entropy loss against buggy statement labels.
        """
        self.model.train()
        
        # Accumulate training samples
        x_train = []
        y_train = []
        
        for df, gt in zip(scores_list, ground_truth_list):
            gt_set = set(gt)
            for _, row in df.iterrows():
                feat = [row['exec_norm'], row['struct_norm'], row['semantic_norm']]
                label = 1.0 if row['statement_id'] in gt_set else 0.0
                x_train.append(feat)
                y_train.append([label])
                
        if not x_train:
            return
            
        x_tensor = torch.FloatTensor(x_train)
        y_tensor = torch.FloatTensor(y_train)
        
        # Train for multiple epochs
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            preds = self.model(x_tensor)
            loss = self.criterion(preds, y_tensor)
            loss.backward()
            self.optimizer.step()
            
    def predict(self, df_scores):
        """
        Predict suspiciousness scores and return sorted DataFrame.
        """
        self.model.eval()
        df = df_scores.copy()
        
        features = np.array(df[['exec_norm', 'struct_norm', 'semantic_norm']].values, dtype=np.float32)
        with torch.no_grad():
            preds = self.model(torch.FloatTensor(features)).numpy().flatten()
            
        df['fusion_score'] = preds
        return df[['statement_id', 'fusion_score']].sort_values(by='fusion_score', ascending=False).reset_index(drop=True)

# =====================================================================
# 2. Learning-to-Rank (RankNet) Baseline
# =====================================================================
class RankNet(nn.Module):
    """
    RankNet network architecture for pairwise Learning-to-Rank.
    """
    def __init__(self, input_dim=3):
        super(RankNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class RankNetBaseline:
    """
    Pairwise RankNet Learning-to-Rank Baseline.
    """
    def __init__(self, lr=0.005, epochs=15, num_pairs_per_bug=50):
        self.epochs = epochs
        self.num_pairs_per_bug = num_pairs_per_bug
        self.model = RankNet(input_dim=3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def fit(self, scores_list, ground_truth_list):
        """
        Train the RankNet model using pairwise cross-entropy loss.
        """
        self.model.train()
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            
            x_i_list = [] # Feature of buggy statements
            x_j_list = [] # Feature of clean statements
            
            for df, gt in zip(scores_list, ground_truth_list):
                gt_set = set(gt)
                
                # Separate buggy and clean statement rows
                buggy_df = df[df['statement_id'].isin(gt_set)]
                clean_df = df[~df['statement_id'].isin(gt_set)]
                
                if buggy_df.empty or clean_df.empty:
                    continue
                    
                buggy_feats = buggy_df[['exec_norm', 'struct_norm', 'semantic_norm']].values
                clean_feats = clean_df[['exec_norm', 'struct_norm', 'semantic_norm']].values
                
                # Generate random pairwise comparisons (buggy statement is preferred over clean)
                for _ in range(self.num_pairs_per_bug):
                    idx_i = random.randint(0, len(buggy_feats) - 1)
                    idx_j = random.randint(0, len(clean_feats) - 1)
                    
                    x_i_list.append(buggy_feats[idx_i])
                    x_j_list.append(clean_feats[idx_j])
                    
            if not x_i_list:
                continue
                
            x_i_tensor = torch.FloatTensor(np.array(x_i_list))
            x_j_tensor = torch.FloatTensor(np.array(x_j_list))
            
            # Forward pass: compute scores
            scores_i = self.model(x_i_tensor)
            scores_j = self.model(x_j_tensor)
            
            # RankNet pairwise loss: push scores_i above scores_j
            # Binary Cross Entropy with target = 1.0 for difference (scores_i - scores_j)
            loss = self.criterion(scores_i - scores_j, torch.ones_like(scores_i))
            
            loss.backward()
            self.optimizer.step()
            
    def predict(self, df_scores):
        """
        Predict relative rankings and return sorted DataFrame.
        """
        self.model.eval()
        df = df_scores.copy()
        
        features = np.array(df[['exec_norm', 'struct_norm', 'semantic_norm']].values, dtype=np.float32)
        with torch.no_grad():
            preds = self.model(torch.FloatTensor(features)).numpy().flatten()
            
        df['fusion_score'] = preds
        return df[['statement_id', 'fusion_score']].sort_values(by='fusion_score', ascending=False).reset_index(drop=True)

# =====================================================================
# 3. Isolated & Formula-based Baselines
# =====================================================================
def get_ochiai_baseline(df_scores):
    """
    Ochiai SBFL Baseline.
    Uses 'exec_norm' directly as the suspiciousness score (assumes Ochiai-normalized execution spectrum).
    """
    df = df_scores.copy()
    df['fusion_score'] = df['exec_norm']
    return df[['statement_id', 'fusion_score']].sort_values(by='fusion_score', ascending=False).reset_index(drop=True)

def get_tarantula_baseline(df_scores):
    """
    Tarantula SBFL Baseline.
    Looks up 'tarantula_norm' if present; otherwise, falls back to 'exec_norm'.
    """
    df = df_scores.copy()
    col = 'tarantula_norm' if 'tarantula_norm' in df.columns else 'exec_norm'
    df['fusion_score'] = df[col]
    return df[['statement_id', 'fusion_score']].sort_values(by='fusion_score', ascending=False).reset_index(drop=True)

def get_dstar_baseline(df_scores):
    """
    DStar SBFL Baseline.
    Looks up 'dstar_norm' if present; otherwise, falls back to 'exec_norm'.
    """
    df = df_scores.copy()
    col = 'dstar_norm' if 'dstar_norm' in df.columns else 'exec_norm'
    df['fusion_score'] = df[col]
    return df[['statement_id', 'fusion_score']].sort_values(by='fusion_score', ascending=False).reset_index(drop=True)

def get_graph_baseline(df_scores):
    """
    Graph-based Structural Dependency Baseline.
    Uses structural score 'struct_norm' only.
    """
    df = df_scores.copy()
    df['fusion_score'] = df['struct_norm']
    return df[['statement_id', 'fusion_score']].sort_values(by='fusion_score', ascending=False).reset_index(drop=True)

def get_transformer_baseline(df_scores):
    """
    Transformer-based Semantic Similarity Baseline.
    Uses semantic score 'semantic_norm' only.
    """
    df = df_scores.copy()
    df['fusion_score'] = df['semantic_norm']
    return df[['statement_id', 'fusion_score']].sort_values(by='fusion_score', ascending=False).reset_index(drop=True)
