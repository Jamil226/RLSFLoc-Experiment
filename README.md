# RLSFLoc: Reinforcement Learning-Based Adaptive Fusion for Fault Localization in Scalable Software Systems

Automated fault localization is a critical software engineering challenge designed to reduce manual debugging effort and accelerate bug-fixing workflows. Traditional Spectrum-Based Fault Localization (SBFL) methods rely solely on code coverage and fail to capture semantic meaning or structural dependencies. 

**RLSFLoc** addresses these limitations by adaptively fusing execution, structural, and semantic evidence using a Reinforcement Learning (RL) agent. By modeling fault localization as a ranking-oriented decision-making problem, the PPO reinforcement learning agent learns to dynamically assign optimal weights to diverse evidence sources based on codebase characteristics.

---

## 🚀 Key Features and Experimental Outcomes

*   **Adaptive PPO Fusion Agent:** Uses a Proximal Policy Optimization (PPO) agent to map a 9D contextual codebase state vector into optimal 3D simplex fusion weights ($\lambda_1 + \lambda_2 + \lambda_3 = 1.0$).
*   **State-of-the-Art Localization Accuracy:** Verified evaluations on the Defects4J benchmark show RLSFLoc achieves:
    *   **Top-1 Accuracy:** **0.80** (presents the fault in the first rank for 80% of the bugs)
    *   **Top-3 / Top-5 / Top-10 Accuracy:** **1.00**
    *   **Mean Reciprocal Rank (MRR):** **0.8917**
    *   **Mean Average Precision (MAP):** **0.8917**
    *   **EXAM Score:** **0.0230** (developers inspect only **2.30%** of the codebase to locate the fault)
*   **Multi-Granular Dependency Extraction:** Automatically parses Java ASTs to extract File, Method, and Statement-level nodes and control-flow, data-flow (def-use chains), call, and inheritance edges.
*   **Contextual Semantic Embeddings:** Encodes tokenized source code and bug descriptions into a shared 384-dimensional vector space using the `all-MiniLM-L6-v2` SentenceTransformer to compute cosine similarities.
*   **Robust Generalizability:** Generalizes well to unseen projects in cross-project settings, maintaining a strong MRR of **0.7431** and an EXAM score of **0.0384**.

---

## 📁 Repository Directory Structure

The project is structured as follows:

```text
├── data/
│   ├── raw/                 # Raw Defects4J splits and AEEEM projects
│   └── processed/           # Processed datasets (aeeem_cleaned.csv, defects4j_ranking.csv)
├── checkpoints/             # Trained model weights (best_actor_model.pth, dqn_fl_model.pth)
├── figures/                 # Copied high-resolution publication plots and graphs
├── results/                 # Raw experimental outputs, significance CSVs, and plots
├── src/                     # Main source code directory
│   ├── agent/               # RL agent architectures
│   │   ├── ppo.py           # Core Proximal Policy Optimization (PPO) agent (Best Actor)
│   │   ├── actor_critic.py  # Actor-Critic network models
│   │   ├── dqn_discrete.py  # DQN agent with discrete weight actions
│   │   └── dqn.py           # Standard DQN architecture
│   ├── environment/         # Custom reinforcement learning environments
│   │   ├── rl_env_gymnasium.py # Aligned Gymnasium environment with 9D context and simplex actions
│   │   └── fl_env.py        # Classical Gym environment
│   ├── model/               # Neural ranking networks
│   │   └── ranker.py        # Neural pairwise ranker implementation
│   ├── utils/               # Modular utility functions and localization processors
│   │   ├── sbfl.py          # Vectorized SBFL formulas (Ochiai, Tarantula, DStar) and normalizers
│   │   ├── structural_extractor.py # Java parser generating program dependency graphs (CFG, call trees)
│   │   ├── propagation.py   # Dependency-aware suspicion propagator and alpha sweep tuner
│   │   ├── semantic_preprocessor.py # CamelCase splitters, comment association, AST parser
│   │   ├── semantic_embedder.py # Lazy-loaded SentenceTransformer semantic similarity encoder
│   │   ├── normalization.py # Aligned multi-score min-max normalizing engine
│   │   ├── fusion.py        # Linear fusion formula applying simplex weights
│   │   ├── metrics.py       # Metrics evaluator (Top-k, MRR, MAP, EXAM, Runtime, Memory)
│   │   ├── ranking_engine.py# Fault ranking report generator
│   │   └── baselines.py     # Comparative baselines (DeepFL MLP, RankNet LTR)
│   ├── evaluate.py          # Core evaluation driver
│   ├── evaluate_baselines.py# side-by-side benchmark evaluation script comparing 8 methods
│   ├── sensitivity_analysis.py # Parametric sweep sweeps LR, discount gamma, epochs, alpha
│   ├── statistical_validation.py # Wilcoxon and Cliff's Delta hypothesis validators
│   ├── generate_publication_figures.py # Plots all publication charts in results/
│   └── train_agents.py      # Trains different RL agent variants
├── tests/                   # 40-case unit test suite verifying every individual module
├── main.tex                 # Academic research manuscript LaTeX source
├── references.bib           # Academic bibliography references database
├── requirements.txt         # Project software requirements
└── README.md                # This file
```

---

## 🛠️ Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Jamil226/RLSFLoc-Experiment.git
   cd RLSFLoc-Experiment
   ```

2. **Set up Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Replication and Execution Commands

To replicate all experimental metrics and validation plots reported in the paper:

### 1. Run Baseline Evaluations
Compare RLSFLoc (PPO) side-by-side with Ochiai, Tarantula, DStar, Graph-Only, Transformer-Only, DeepFL-like MLP, and RankNet LTR:
```bash
PYTHONPATH=. python src/evaluate_baselines.py
```
*Outputs a detailed comparative analysis report in terminal covering Top-k, MRR, MAP, EXAM, Runtime, and Peak Memory.*

### 2. Run Hyperparameter Sensitivity Sweep
Sweep structural propagation coefficient $\alpha \in [0.0, 1.0]$, discount factor $\gamma$, training epochs, and PPO learning rate:
```bash
PYTHONPATH=. python src/sensitivity_analysis.py
```
*Outputs optimal grid parameters and saves plots in `results/alpha_sensitivity_plot.png` and `results/lr_sensitivity_plot.png`.*

### 3. Run Statistical Hypothesis Testing
Verify statistical validation of RLSFLoc against all baselines using Wilcoxon signed-rank and Cliff's Delta effect size tests:
```bash
PYTHONPATH=. python src/statistical_validation.py
```
*Saves statistical significance boxplots in `results/significance_plot.png` and `results/exam_significance_plot.png`.*

### 4. Generate Publication Figures
Regenerate all 8 high-resolution publication charts:
```bash
PYTHONPATH=. python src/generate_publication_figures.py
```
*Stores all publication figures in the `results/` folder.*

### 5. Verify via Unit Tests
Execute the comprehensive 40-case unit test suite to verify module calculations, Astro preprocessors, and environment actions:
```bash
PYTHONPATH=. pytest tests/ -v -s
```

---

## 📊 Scientific Framework and Formulations

The mathematical framework powering RLSFLoc is fully aligned with implementation code:

1. **Ochiai suspiciousness ($S_{exec}$):**
   $$S_{exec}(v_i) = \frac{n_{ef}(v_i)}{\sqrt{|\mathcal{T}_f| \cdot (n_{ef}(v_i)+n_{ep}(v_i))}}$$
2. **Structural Graph Propagation ($S_{struct}$):**
   $$S_{struct}(v_i) = (1-\alpha)S_{exec}(v_i) + \alpha \sum_{v_j \in N(v_i)} w_{ij}S_{exec}(v_j)$$
   *Subject to normalizations $\sum_{v_j \in N(v_i)} w_{ij} = 1.0$, with optimal propagation alpha $\alpha = 0.60$.*
3. **Semantic Relevance similarity ($S_{sem}$):**
   $$S_{sem}(v_i) = \frac{\mathbf{z}_i^T\mathbf{z}_B}{\|\mathbf{z}_i\|_2\|\mathbf{z}_B\|_2}$$
4. **Simplex Fusion Score ($S_{fusion}$):**
   $$S_{fusion}(v_i) = \lambda_1\hat{S}_{exec}(v_i) + \lambda_2\hat{S}_{struct}(v_i) + \lambda_3\hat{S}_{sem}(v_i)$$
   *Subject to simplex constraints $\sum \lambda_j = 1.0$ dynamically computed by the PPO actor policy.*

---

## 📬 Open-Source Replication and Citation

The source code, environment, and datasets are publicly hosted at:
[https://github.com/Jamil226/RLSFLoc-Experiment](https://github.com/Jamil226/RLSFLoc-Experiment)

If you use RLSFLoc in your software engineering research, please cite our corresponding publication:

```bibtex
@article{jamil2026rlsfloc,
  author    = {Muhammad Jamil and Adnan Kavak and Sema Bayraktar and Muhammad Farhan and Gautam Srivastava and Hossein Fotouhi},
  title     = {RLSFLoc: Reinforcement Learning-Based Adaptive Fusion for Fault Localization in Scalable Software Systems},
  journal   = {Special Issue on AI/ML in Software Engineering},
  year      = {2026}
}
```
