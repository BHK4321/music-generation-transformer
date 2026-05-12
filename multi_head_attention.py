import torch
import torch.nn as nn
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, dropout: nn.Dropout = None) -> torch.Tensor:
        d_k = query.shape[-1]
        # (Batch_size, num_heads, Seq_len, d_k) @ (Batch_size, num_heads, d_k, Seq_len) -> (Batch_size, num_heads, Seq_len, Seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = torch.softmax(attention_scores, dim=-1) # (Batch_size, num_heads, Seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return torch.matmul(attention_scores, value), attention_scores # (Batch_size, num_heads, Seq_len, d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.w_q(query) # (Batch_size, Seq_len, d_model)
        key = self.w_k(key) # (Batch_size, Seq_len, d_model)
        value = self.w_v(value) # (Batch_size, Seq_len, d_model)

        query = query.view(query.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) # (Batch_size, num_heads, Seq_len, d_k)
        key = key.view(key.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) # (Batch_size, num_heads, Seq_len, d_k)
        value = value.view(value.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) # (Batch_size, num_heads, Seq_len, d_k)

        if past_key is not None:
            key = torch.cat([past_key, key], dim=-2)
        if past_value is not None:
            value = torch.cat([past_value, value], dim=-2)

        x, self.attention_scores = self.attention(query, key, value, mask=mask, dropout=self.dropout) # (Batch_size, num_heads, Seq_len, d_k)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model) # (Batch_size, Seq_len, d_model)
        output = self.w_o(x)

        if use_cache:
            return output, key, value

        return output # (Batch_size, Seq_len, d_model)