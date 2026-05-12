import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
 
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative")

        end_pos = start_pos + x.size(1)
        if end_pos > self.pe.size(1):
            raise ValueError(
                f"Requested positions up to {end_pos}, but maximum supported is {self.pe.size(1)}"
            )

        x = x + (self.pe[:, start_pos:end_pos, :]).requires_grad_(False)
        x = self.dropout(x)
        return x 