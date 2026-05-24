# RLSFLoc-Experiment

Reinforcement Learning and Supervised Fault Localization Framework.

## Directory Structure
- `data/`: Contains raw and preprocessed datasets.
  - `raw/defects4j/`: Original Defects4J splits.
  - `raw/aeeem/`: Original AEEEM data.
  - `processed/`: Processed training metrics (`aeeem_cleaned.csv` and `defects4j_ranking.csv`).
- `src/`: Main source files for modeling, training, and evaluation.
  - `environment/`: Custom Gym fault localization environment (`fl_env.py`).
  - `agent/`: RL agents (DQN agent in `dqn.py`, Policy Gradient agent in `policy_gradient.py`).
  - `model/`: Neural representation/ranking networks (`ranker.py`).
  - `utils/`: Data loaders and metric score evaluators.
- `tests/`: Project tests.
- `checkpoints/`: Model checkpoints saved during training.
- `results/`: Process evaluation tables and logs.

## Setup
Install the dependencies:
```bash
pip install -r requirements.txt
```

## Running
To train the RLSFLoc DQN Agent:
```bash
python src/train.py
```

To evaluate the trained DQN Agent:
```bash
python src/evaluate.py
```
