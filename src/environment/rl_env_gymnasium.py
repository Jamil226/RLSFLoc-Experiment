import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class RLSFLocEnv(gym.Env):
    """
    A custom Gymnasium environment for Reinforcement Learning-based Fault Localization (RLSFLoc).
    
    This environment is designed as a contextual optimization task:
      - State (Context): A 9D vector representing codebase/bug stats:
        [mean, std, max] for each of [exec_norm, struct_norm, semantic_norm]
      - Action: A continuous 3D vector which is projected onto the simplex:
        [lambda1, lambda2, lambda3] subject to sum(lambda) = 1, lambda >= 0 (using Softmax)
      - Reward: A unified reward reflecting ranking quality improvement over the raw execution-only
        baseline across three key metrics:
        1. Top-k Improvement
        2. MRR (Mean Reciprocal Rank) Improvement
        3. EXAM Score Reduction (reduction in developer inspection effort)
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, normalized_scores_list, ground_truth_list, k=5, reward_weights=None):
        """
        Parameters:
        -----------
        normalized_scores_list : list of pandas.DataFrame
            List of DataFrames, where each represents a bug/codebase and contains columns:
            ['statement_id', 'exec_norm', 'struct_norm', 'semantic_norm']
        ground_truth_list : list of (list or set)
            List of sets containing actual faulty statement IDs for each corresponding bug.
        k : int, optional (default=5)
            The threshold for the Top-k accuracy metric.
        reward_weights : dict, optional
            Coefficients for the reward function.
            Default: {"top_k": 1.0, "mrr": 2.0, "exam": 5.0}
        """
        super(RLSFLocEnv, self).__init__()
        
        self.normalized_scores_list = normalized_scores_list
        self.ground_truth_list = [set(gt) for gt in ground_truth_list]
        self.k = k
        
        if reward_weights is None:
            self.reward_weights = {"top_k": 1.0, "mrr": 2.0, "exam": 5.0}
        else:
            self.reward_weights = reward_weights

        if len(self.normalized_scores_list) != len(self.ground_truth_list):
            raise ValueError("Dimension mismatch: normalized_scores_list and ground_truth_list must have the same length")
            
        if not self.normalized_scores_list:
            raise ValueError("normalized_scores_list cannot be empty")

        # Action Space: Continuous real-valued 3D vector.
        # Softmax is applied to project this vector onto a valid 3D simplex.
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        # Observation Space: A 9D vector representing the bug state/context:
        # [mean_exec, std_exec, max_exec, mean_struct, std_struct, max_struct, mean_sem, std_sem, max_sem]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )

        self.current_episode = 0
        self.current_idx = 0
        self.current_df = None
        self.current_gt = None

    def _get_observation(self):
        """
        Computes the 9D context/observation state for the current codebase.
        """
        df = self.current_df
        
        # Calculate statistics
        mean_exec = float(df['exec_norm'].mean())
        std_exec = float(df['exec_norm'].std()) if len(df) > 1 else 0.0
        max_exec = float(df['exec_norm'].max())
        
        mean_struct = float(df['struct_norm'].mean())
        std_struct = float(df['struct_norm'].std()) if len(df) > 1 else 0.0
        max_struct = float(df['struct_norm'].max())
        
        mean_sem = float(df['semantic_norm'].mean())
        std_sem = float(df['semantic_norm'].std()) if len(df) > 1 else 0.0
        max_sem = float(df['semantic_norm'].max())
        
        obs = np.array([
            mean_exec, std_exec, max_exec,
            mean_struct, std_struct, max_struct,
            mean_sem, std_sem, max_sem
        ], dtype=np.float32)
        
        # Replace NaNs/Infs with 0.0 just in case
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new episode/bug context.
        """
        super().reset(seed=seed)
        
        # Choose the bug/codebase index
        if options and 'index' in options:
            self.current_idx = int(options['index']) % len(self.normalized_scores_list)
        else:
            self.current_idx = self.current_episode % len(self.normalized_scores_list)
            
        self.current_df = self.normalized_scores_list[self.current_idx]
        self.current_gt = self.ground_truth_list[self.current_idx]
        
        obs = self._get_observation()
        
        info = {
            "index": self.current_idx,
            "num_statements": len(self.current_df),
            "num_faults": len(self.current_gt)
        }
        
        return obs, info

    def step(self, action):
        """
        Executes a step in the environment by applying the fusion weights,
        calculating metrics improvements, and returning rewards.
        
        This environment is single-step (contextual bandit), so terminated=True.
        """
        # 1. Project raw action onto the valid 3D simplex using Softmax
        # Prevents division by zero or negative weights natively
        exp_action = np.exp(action - np.max(action)) # Subtract max for numerical stability
        lambdas = exp_action / (np.sum(exp_action) + 1e-12)
        
        l1, l2, l3 = lambdas
        
        # 2. Compute Fused suspiciousness score (Section 3.4 Equation 496)
        # S_fusion = l1 * exec_norm + l2 * struct_norm + l3 * semantic_norm
        fused_scores = (
            l1 * self.current_df['exec_norm'] +
            l2 * self.current_df['struct_norm'] +
            l3 * self.current_df['semantic_norm']
        )
        
        # Create ranking DataFrame for fused scores
        rank_df = pd.DataFrame({
            'statement_id': self.current_df['statement_id'].values,
            'exec_score': self.current_df['exec_norm'].values,
            'fused_score': fused_scores.values
        })
        
        # Rank both ascending=False. method='min' handles ties robustly.
        rank_df['exec_rank'] = rank_df['exec_score'].rank(ascending=False, method='min')
        rank_df['fused_rank'] = rank_df['fused_score'].rank(ascending=False, method='min')
        
        num_statements = len(rank_df)
        
        # 3. Calculate rank improvement metrics across all ground truth faults
        top_k_improvements = []
        mrr_improvements = []
        exam_reductions = []
        
        fused_ranks = []
        exec_ranks = []
        
        for fault in self.current_gt:
            if fault in rank_df['statement_id'].values:
                f_idx = rank_df[rank_df['statement_id'] == fault].index[0]
                r_exec = rank_df.loc[f_idx, 'exec_rank']
                r_fused = rank_df.loc[f_idx, 'fused_rank']
            else:
                # If a ground truth fault was not mapped, penalize with max rank
                r_exec = num_statements
                r_fused = num_statements
                
            fused_ranks.append(r_fused)
            exec_ranks.append(r_exec)
            
            # A. Top-k Improvement:
            # Positive reward if pushed into Top-k, negative if pushed out
            hit_exec = 1.0 if r_exec <= self.k else 0.0
            hit_fused = 1.0 if r_fused <= self.k else 0.0
            top_k_improvements.append(hit_fused - hit_exec)
            
            # B. MRR (Mean Reciprocal Rank) Improvement:
            mrr_improvements.append((1.0 / r_fused) - (1.0 / r_exec))
            
            # C. EXAM Reduction (inspect fewer statements):
            exam_reductions.append((r_exec / num_statements) - (r_fused / num_statements))

        # Average metrics across all faults for this bug
        avg_top_k_imp = np.mean(top_k_improvements) if top_k_improvements else 0.0
        avg_mrr_imp = np.mean(mrr_improvements) if mrr_improvements else 0.0
        avg_exam_red = np.mean(exam_reductions) if exam_reductions else 0.0
        
        # 4. Compute Unified Reward
        reward = (
            self.reward_weights['top_k'] * avg_top_k_imp +
            self.reward_weights['mrr'] * avg_mrr_imp +
            self.reward_weights['exam'] * avg_exam_red
        )
        
        # 5. Terminate episode immediately (Contextual Bandit design)
        terminated = True
        truncated = False
        
        obs = self._get_observation()
        
        info = {
            "lambda1": float(l1),
            "lambda2": float(l2),
            "lambda3": float(l3),
            "avg_baseline_rank": float(np.mean(exec_ranks)) if exec_ranks else 0.0,
            "avg_fused_rank": float(np.mean(fused_ranks)) if fused_ranks else 0.0,
            "top_k_improvement": float(avg_top_k_imp),
            "mrr_improvement": float(avg_mrr_imp),
            "exam_reduction": float(avg_exam_red)
        }
        
        self.current_episode += 1
        
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
