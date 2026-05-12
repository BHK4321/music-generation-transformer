import torch
import torch.nn as nn
from layer_norm import LayerNorm

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float, d_model: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(sublayer_output(self.norm(x)))