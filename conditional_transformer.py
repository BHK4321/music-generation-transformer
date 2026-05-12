from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from decoder import Decoder, DecoderLayer
from feed_forward import FeedForward
from input_embedding import InputEmbeddings
from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding
from projection_layer import ProjectionLayer
from transformer import GPTmini


class ConditionalGPTmini(GPTmini):
    def __init__(
        self,
        decoder: Decoder,
        projection_layer: ProjectionLayer,
        token_embed: InputEmbeddings,
        token_pos: PositionalEncoding,
        context_projection: Optional[nn.Module] = None,
    ):
        super().__init__(decoder=decoder, projection_layer=projection_layer, token_embed=token_embed, token_pos=token_pos)
        self.context_projection = context_projection
        self.model_dim = decoder.layers[0].self_attention.d_model if len(decoder.layers) > 0 else token_embed.d_model

    def _project_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None:
            return None
        if context.size(-1) == self.model_dim:
            return context
        if self.context_projection is None:
            return context
        return self.context_projection(context)

    def forward(
        self,
        tokens: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        return super().forward(
            tokens,
            tgt_mask=tgt_mask,
            start_pos=start_pos,
            context=self._project_context(context),
            context_mask=context_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = False,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            context=self._project_context(context),
            context_mask=context_mask,
        )


def build_conditional_model(
    vocab_size: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int,
    max_seq_len: int,
    dropout: float,
    device: torch.device,
    context_hidden_size: Optional[int] = None,
) -> ConditionalGPTmini:
    layers = []
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        layers.append(
            DecoderLayer(
                self_attention=self_attn,
                feed_forward=ff,
                dropout=dropout,
                cross_attention=cross_attn,
            )
        )

    decoder = Decoder(layers=nn.ModuleList(layers), d_model=d_model)
    projection = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)
    token_embed = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
    token_pos = PositionalEncoding(d_model=d_model, seq_len=max_seq_len, dropout=dropout)

    context_projection: Optional[nn.Module] = None
    if context_hidden_size is not None and context_hidden_size != d_model:
        context_projection = nn.Sequential(
            nn.Linear(context_hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    model = ConditionalGPTmini(
        decoder=decoder,
        projection_layer=projection,
        token_embed=token_embed,
        token_pos=token_pos,
        context_projection=context_projection,
    ).to(device)
    model.eval()
    return model


def load_partial_checkpoint(model: nn.Module, checkpoint_path: str, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint_state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        checkpoint_state = checkpoint
    else:
        raise ValueError("Checkpoint must be a state dict or a dict containing 'model_state_dict'")

    current_state = model.state_dict()
    loaded_keys = {}
    skipped_keys = []

    for key, value in checkpoint_state.items():
        if key in current_state and current_state[key].shape == value.shape:
            loaded_keys[key] = value
        else:
            skipped_keys.append(key)

    current_state.update(loaded_keys)
    model.load_state_dict(current_state)
    return {
        "loaded_key_count": len(loaded_keys),
        "skipped_key_count": len(skipped_keys),
        "skipped_keys": skipped_keys,
    }


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True