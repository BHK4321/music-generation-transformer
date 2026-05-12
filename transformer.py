import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from decoder import Decoder
from projection_layer import ProjectionLayer
from input_embedding import InputEmbeddings
from positional_encoding import PositionalEncoding

class GPTmini(nn.Module):
    def __init__(
        self,
        decoder: Decoder,
        projection_layer: ProjectionLayer,
        token_embed: InputEmbeddings,
        token_pos: PositionalEncoding,
    ):
        super().__init__()
        self.decoder = decoder
        self.projection_layer = projection_layer
        self.token_embed = token_embed
        self.token_pos = token_pos

    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    def decode(
        self,
        tokens: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        if tgt_mask is None and not use_cache:
            tgt_mask = self.create_causal_mask(tokens.size(1), tokens.device)

        x = self.token_embed(tokens)
        x = self.token_pos(x, start_pos=start_pos)
        return self.decoder(
            x,
            tgt_mask,
            context=context,
            context_mask=context_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x)

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
        if tgt_mask is None and not use_cache:
            tgt_mask = self.create_causal_mask(tokens.size(1), tokens.device)

        x = self.token_embed(tokens)
        x = self.token_pos(x, start_pos=start_pos)
        decoder_result = self.decoder(
            x,
            tgt_mask,
            context=context,
            context_mask=context_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )

        if use_cache:
            dec_output, present_kv = decoder_result
            logits = self.projection_layer(dec_output)
            return logits, present_kv

        dec_output = decoder_result
        return self.projection_layer(dec_output)

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
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if max_new_tokens <= 0:
            return tokens

        was_training = self.training
        self.eval()

        generated = tokens
        max_positions = self.token_pos.pe.size(1)
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        if use_cache:
            if generated.size(1) > max_positions:
                raise ValueError(
                    f"Sequence length {generated.size(1)} exceeds positional encoding limit {max_positions}"
                )
            prefill_mask = self.create_causal_mask(generated.size(1), generated.device)
            logits_or_log_probs, past_kv = self.forward(
                generated,
                tgt_mask=prefill_mask,
                context=context,
                context_mask=context_mask,
                use_cache=True,
            )
            next_token_scores = logits_or_log_probs[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, next_token_scores.size(-1))
                threshold = torch.topk(next_token_scores, k=k, dim=-1).values[:, -1].unsqueeze(-1)
                next_token_scores = next_token_scores.masked_fill(next_token_scores < threshold, float("-inf"))

            if do_sample:
                probs = torch.softmax(next_token_scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_scores, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                if was_training:
                    self.train()
                return generated

            remaining_steps = max_new_tokens - 1
            for _ in range(remaining_steps):
                if generated.size(1) > max_positions:
                    raise ValueError(
                        f"Sequence length {generated.size(1)} exceeds positional encoding limit {max_positions}"
                    )

                input_token = generated[:, -1:]
                start_pos = generated.size(1) - 1
                logits_or_log_probs, past_kv = self.forward(
                    input_token,
                    tgt_mask=None,
                    start_pos=start_pos,
                    context=context,
                    context_mask=context_mask,
                    past_kv=past_kv,
                    use_cache=True,
                )
                next_token_scores = logits_or_log_probs[:, -1, :] / temperature

                if top_k is not None and top_k > 0:
                    k = min(top_k, next_token_scores.size(-1))
                    threshold = torch.topk(next_token_scores, k=k, dim=-1).values[:, -1].unsqueeze(-1)
                    next_token_scores = next_token_scores.masked_fill(next_token_scores < threshold, float("-inf"))

                if do_sample:
                    probs = torch.softmax(next_token_scores, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_scores, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                    break

            if was_training:
                self.train()
            return generated

        for _ in range(max_new_tokens):
            if generated.size(1) > max_positions:
                raise ValueError(
                    f"Sequence length {generated.size(1)} exceeds positional encoding limit {max_positions}"
                )

            logits_or_log_probs = self.forward(generated, context=context, context_mask=context_mask)
            next_token_scores = logits_or_log_probs[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                top_k = min(top_k, next_token_scores.size(-1))
                threshold = torch.topk(next_token_scores, k=top_k, dim=-1).values[:, -1].unsqueeze(-1)
                next_token_scores = next_token_scores.masked_fill(next_token_scores < threshold, float("-inf"))

            if do_sample:
                probs = torch.softmax(next_token_scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_scores, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break

        if was_training:
            self.train()

        return generated