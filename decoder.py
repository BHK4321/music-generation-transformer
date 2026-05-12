import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from residual_connection import ResidualConnection
from layer_norm import LayerNorm

class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
        cross_attention: Optional[MultiHeadAttention] = None,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        num_residuals = 3 if cross_attention is not None else 2
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout, self_attention.d_model) for _ in range(num_residuals)]
        )

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if use_cache:
            past_key, past_value = (past_kv if past_kv is not None else (None, None))
            norm_x = self.residual_connections[0].norm(x)
            attn_output, present_key, present_value = self.self_attention(
                norm_x,
                norm_x,
                norm_x,
                tgt_mask,
                past_key=past_key,
                past_value=past_value,
                use_cache=True,
            )
            x = x + self.residual_connections[0].dropout(attn_output)
        else:
            x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))

        if self.cross_attention is not None and context is not None:
            x = self.residual_connections[1](x, lambda x: self.cross_attention(x, context, context, context_mask))
            x = self.residual_connections[2](x, self.feed_forward)
        else:
            x = self.residual_connections[1](x, self.feed_forward)

        if use_cache:
            return x, (present_key, present_value)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        present_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx, layer in enumerate(self.layers):
            layer_past = None if past_kv is None else past_kv[layer_idx]
            if use_cache:
                x, layer_present = layer(
                    x,
                    tgt_mask,
                    context=context,
                    context_mask=context_mask,
                    past_kv=layer_past,
                    use_cache=True,
                )
                present_kv.append(layer_present)
            else:
                x = layer(x, tgt_mask, context=context, context_mask=context_mask)

        x = self.norm(x)
        if use_cache:
            return x, present_kv
        return x
