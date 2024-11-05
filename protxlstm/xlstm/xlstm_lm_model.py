# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck

# Modified by Pieter-Jan Hoedt, Niklas Schmidinger, Lisa Schneckenreiter and Sohvi Luukkonen 
#   - Remove sLSTM
#   - Add positional embeddings
#   - Modify forward to take and return state


from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .components.init import small_init_init_
from .components.rotary_position import compute_freqs_cis
from .utils import WeightDecayOptimGroupMixin
from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig


@dataclass
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False
    position_embeddings: str = "none"
    max_position_embeddings: int = 2048
    max_seq_position_embeddings: int = 512
    rope_base_frequency: int = 10_000


class xLSTMLMModel(WeightDecayOptimGroupMixin, nn.Module):
    config_class = xLSTMLMModelConfig

    def __init__(self, config: xLSTMLMModelConfig, **kwargs):
        super().__init__()
        self.config = config

        self.xlstm_block_stack = xLSTMBlockStack(config=config)

        assert config.position_embeddings in [
            "abs_1d",
            "abs_2d",
            "rot",
            "rot_1d",
            "rot_2d",
            "none",
        ], f"Unknown position embeddings: {config.position_embeddings}"

        if config.position_embeddings == "abs_1d":
            assert (
                config.embedding_dim % 2 == 0
            ), "for abs_1d embedding_dim must be divisible by 2."
            self.token_embedding = nn.Embedding(
                config.vocab_size, config.embedding_dim // 2
            )
            self.position_embedding = nn.Embedding(
                config.max_position_embeddings,
                config.embedding_dim - config.embedding_dim // 2,
            )
        elif config.position_embeddings == "abs_2d":
            assert (
                config.embedding_dim % 4 == 0
            ), "for abs_1d embedding_dim must be divisible by 4."
            self.token_embedding = nn.Embedding(
                config.vocab_size, config.embedding_dim - 2 * config.embedding_dim // 4
            )
            self.position_embedding = nn.Embedding(
                config.max_position_embeddings, config.embedding_dim // 4
            )
            self.seq_position_embedding = nn.Embedding(
                config.max_seq_position_embeddings, config.embedding_dim // 4
            )
        elif config.position_embeddings.startswith("rot"):

            head_dim = config.mlstm_block.mlstm._inner_embedding_dim
            assert head_dim % 2 == 0, "RoPE requires even head dimension"
            self.token_embedding = nn.Embedding(
                num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim
            )

            if config.position_embeddings == "rot":
                max_positions = config.max_position_embeddings * config.max_seq_position_embeddings
                freqs_cos, freqs_sin = compute_freqs_cis(torch.arange(max_positions), head_dim)
                self.register_buffer("freqs_cos", freqs_cos, persistent=False)
                self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        else:
            self.token_embedding = nn.Embedding(
                num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim
            )

        self.emb_dropout = (
            nn.Dropout(config.dropout)
            if config.add_embedding_dropout
            else nn.Identity()
        )

        self.lm_head = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            bias=False,
        )
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight


    def reset_parameters(self):
        self.xlstm_block_stack.reset_parameters()

        small_init_init_(
            self.token_embedding.weight, dim=self.token_embedding.embedding_dim
        )

        if not self.config.tie_weights:
            small_init_init_(self.lm_head.weight, dim=self.config.embedding_dim)

        if hasattr(self, "position_embedding"):
            small_init_init_(
                self.position_embedding.weight, dim=self.position_embedding.embedding_dim
            )

        if hasattr(self, "seq_position_embedding"):
            small_init_init_(self.seq_position_embedding.weight, dim=self.seq_position_embedding.embedding_dim)

    def forward(self, input_ids: torch.Tensor, state=None, **kwargs) -> torch.Tensor:

        x = self.token_embedding(input_ids)

        # absolute position embeddings
        if self.config.position_embeddings.startswith("abs"):
            position_ids = kwargs.pop("position_ids", None)
            position_embeddings = self.position_embedding(position_ids)

            seq_position_ids = kwargs.pop("seq_position_ids", None)  # check if abs_2d
            if seq_position_ids is not None:
                seq_position_embeddings = self.seq_position_embedding(seq_position_ids)
                position_embeddings = torch.cat(
                    [position_embeddings, seq_position_embeddings], dim=-1
                )

            x = torch.cat([x, position_embeddings], dim=-1)

        # rotary postion embeddings
        elif self.config.position_embeddings.startswith("rot"):
            if self.config.position_embeddings.endswith("1d"):
                assert "position_ids" in kwargs, "1d RoPE requires 'position_ids' argument"
                head_dim = self.config.mlstm_block.mlstm._inner_embedding_dim
                freqs_cos, freqs_sin = compute_freqs_cis(kwargs.pop("position_ids"), head_dim, theta=self.config.rope_base_frequency)
            elif self.config.position_embeddings.endswith("2d"):
                assert (
                    "position_ids" in kwargs and "seq_position_ids" in kwargs
                ), "2d RoPE requires 'position_ids' and 'seq_position_ids' arguments"
                head_dim = self.config.mlstm_block.mlstm._inner_embedding_dim
                total_emb = self.config.max_position_embeddings + self.config.max_seq_position_embeddings
                pos_dim = head_dim * self.config.max_position_embeddings // total_emb
                pos_dim -= pos_dim % 2  # assure pos_dim is even
                seq_dim = head_dim - pos_dim
                freqs_cos1, freqs_sin1 = compute_freqs_cis(kwargs.pop("position_ids"), pos_dim, theta=self.config.rope_base_frequency)
                freqs_cos2, freqs_sin2 = compute_freqs_cis(kwargs.pop("seq_position_ids"), seq_dim, theta=self.config.rope_base_frequency)
                freqs_cos = torch.cat([freqs_cos1, freqs_cos2], dim=-1)
                freqs_sin = torch.cat([freqs_sin1, freqs_sin2], dim=-1)
            else:
                assert hasattr(self, "freqs_cos"), "model was not configured for general RoPE"
                assert len(self.freqs_cos) >= x.shape[1], "input sequence longer than max_seq_positions"
                freqs_cos, freqs_sin = self.freqs_cos[:x.shape[1]], self.freqs_sin[:x.shape[1]]

            kwargs["freqs_cos"] = freqs_cos
            kwargs["freqs_sin"] = freqs_sin

        x = self.emb_dropout(x)

        if self.config.mlstm_block.mlstm.return_last_state:
            x, state = self.xlstm_block_stack(x, state=state, **kwargs)
        else:
            x = self.xlstm_block_stack(x, state=state, **kwargs)
        
        logits = self.lm_head(x)

        if self.config.mlstm_block.mlstm.return_last_state:
            return logits, state
        else:
            return logits


    def step(
        self,
        input_ids: torch.Tensor,
        state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:

        x = self.token_embedding(input_ids)

        # absolute position embeddings
        if self.config.position_embeddings.startswith("abs"):
            position_ids = kwargs.pop("position_ids", None)
            position_embeddings = self.position_embedding(position_ids)

            seq_position_ids = kwargs.pop("seq_position_ids", None)  # check if abs_2d
            if seq_position_ids is not None:
                seq_position_embeddings = self.seq_position_embedding(seq_position_ids)
                position_embeddings = torch.cat(
                    [position_embeddings, seq_position_embeddings], dim=-1
                )

            x = torch.cat([x, position_embeddings], dim=-1)

        # rotary postion embeddings
        elif self.config.position_embeddings.startswith("rot"):
            if self.config.position_embeddings.endswith("1d"):
                assert "position_ids" in kwargs, "1d RoPE requires 'position_ids' argument"
                head_dim = self.config.mlstm_block.mlstm._inner_embedding_dim
                freqs_cos, freqs_sin = compute_freqs_cis(kwargs.pop("position_ids"), head_dim, theta=self.config.rope_base_frequency)
                kwargs.pop("seq_position_ids", None)

            elif self.config.position_embeddings.endswith("2d"):
                assert (
                    "position_ids" in kwargs and "seq_position_ids" in kwargs
                ), "2d RoPE requires 'position_ids' and 'seq_position_ids' arguments"
                head_dim = self.config.mlstm_block.mlstm._inner_embedding_dim
                total_emb = self.config.max_position_embeddings + self.config.max_seq_position_embeddings
                pos_dim = head_dim * self.config.max_position_embeddings // total_emb
                pos_dim -= pos_dim % 2  # assure pos_dim is even
                seq_dim = head_dim - pos_dim
                freqs_cos1, freqs_sin1 = compute_freqs_cis(kwargs.pop("position_ids"), pos_dim, theta=self.config.rope_base_frequency)
                freqs_cos2, freqs_sin2 = compute_freqs_cis(kwargs.pop("seq_position_ids"), seq_dim, theta=self.config.rope_base_frequency)
                freqs_cos = torch.cat([freqs_cos1, freqs_cos2], dim=-1)
                freqs_sin = torch.cat([freqs_sin1, freqs_sin2], dim=-1)
            else:
                assert hasattr(self, "freqs_cos"), "model was not configured for general RoPE"
                assert len(self.freqs_cos) >= x.shape[1], "input sequence longer than max_seq_positions"
                freqs_cos, freqs_sin = self.freqs_cos[:x.shape[1]], self.freqs_sin[:x.shape[1]]

            kwargs["freqs_cos"] = freqs_cos
            kwargs["freqs_sin"] = freqs_sin

        x = self.emb_dropout(x)
        x, state = self.xlstm_block_stack.step(x, state=state, **kwargs)
        logits = self.lm_head(x)
        return logits, state

    def _create_weight_decay_optim_groups(
        self, **kwargs
    ) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(
            **kwargs
        )
        # remove token embedding and add it to the correct group, accrording to the config
        weight_decay = list(weight_decay)
        removed = 0
        for idx in range(len(weight_decay)):
            if weight_decay[idx - removed] is self.token_embedding.weight:
                weight_decay.pop(idx - removed)
                removed += 1
        weight_decay = tuple(weight_decay)
        if self.config.weight_decay_on_embedding:
            weight_decay += (self.token_embedding.weight,)
        else:
            no_weight_decay += (self.token_embedding.weight,)

        return weight_decay, no_weight_decay
