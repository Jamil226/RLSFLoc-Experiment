import torch
import torch.nn as nn

class CodeRanker(nn.Module):
    """
    Neural Network model designed to score and rank code components
    based on code metrics and textual feature representation.
    """
    def __init__(self, input_dim):
        super(CodeRanker, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
