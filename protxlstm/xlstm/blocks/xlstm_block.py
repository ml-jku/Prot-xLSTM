# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck

# Modified by Pieter-Jan Hoedt, Niklas Schmidinger, Lisa Schneckenreiter and Sohvi Luukkonen 
#   - Remove sLSTM
#   - Modify forward to take and return state


from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..components.feedforward import FeedForwardConfig, create_feedforward
from ..components.ln import LayerNorm
from .mlstm.layer import mLSTMLayer, mLSTMLayerConfig

"""An xLSTM block can be either an sLSTM Block or an mLSTM Block.
In this repository only mLSTM is implemented.

It contains the pre-LayerNorms and the skip connections.
"""


@dataclass
class xLSTMBlockConfig:
    mlstm: Optional[mLSTMLayerConfig] = None

    feedforward: Optional[FeedForwardConfig] = None

    _num_blocks: int = 1
    _block_idx: int = 0

    def __post_init__(self):
        assert (
            self.mlstm is not None
        ), "mlstm config must be provided"

        embedding_dim = (
            self.mlstm.embedding_dim
        )

        self.mlstm._num_blocks = self._num_blocks
        self.mlstm._block_idx = self._block_idx

        if self.feedforward:
            self.feedforward.embedding_dim = embedding_dim
            self.feedforward._num_blocks = self._num_blocks
            self.feedforward.__post_init__()


class xLSTMBlock(nn.Module):

    config_class = xLSTMBlockConfig

    def __init__(self, config: xLSTMBlockConfig) -> None:
        super().__init__()
        self.config = config
        embedding_dim = (
            self.config.mlstm.embedding_dim
        )

        self.xlstm_norm = LayerNorm(ndim=embedding_dim, weight=True, bias=False)

        if self.config.mlstm is not None:
            self.xlstm = mLSTMLayer(config=self.config.mlstm)
        else:
            raise ValueError("mlstm must be provided")

        if self.config.feedforward is not None:
            self.ffn_norm = LayerNorm(
                ndim=self.config.feedforward.embedding_dim, weight=True, bias=False
            )
            self.ffn = create_feedforward(config=self.config.feedforward)
        else:
            self.ffn_norm = None
            self.ffn = None

        self.reset_parameters()

    def forward(self, x: torch.Tensor, state=None, **kwargs) -> torch.Tensor:
        if self.config.mlstm.return_last_state:
            x_xlstm, xlstm_state = self.xlstm(self.xlstm_norm(x), state=state, **kwargs)
            x = x + x_xlstm
        else:
            x = x + self.xlstm(self.xlstm_norm(x), state=state, **kwargs)

        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x), **kwargs)

        if self.config.mlstm.return_last_state:
            return x, xlstm_state
        else:
            return x
    
    def step(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        x_xlstm, xlstm_state = self.xlstm.step(self.xlstm_norm(x), **kwargs)
        x = x + x_xlstm
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x), **kwargs)
        return x, xlstm_state
    
    def reset_parameters(self) -> None:
        
        self.xlstm.reset_parameters()
        self.xlstm_norm.reset_parameters()
    
        if self.ffn is not None:
            self.ffn.reset_parameters()
            self.ffn_norm.reset_parameters()
