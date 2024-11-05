# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck

# Modified by Pieter-Jan Hoedt, Niklas Schmidinger, Lisa Schneckenreiter and Sohvi Luukkonen 
#   - Remove sLSTM
#   - Modify forward to take and return state


from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from .components.ln import LayerNorm


@dataclass
class xLSTMBlockStackConfig:
    mlstm_block: Optional[mLSTMBlockConfig] = None

    context_length: int = -1
    num_blocks: int = 1
    embedding_dim: int = 128
    add_post_blocks_norm: bool = True
    bias: bool = False
    dropout: float = 0.0

    checkpoint_blocks: bool = False

    # _block_map is a string that specifies which block is used at which position
    # 0: use the mLSTM block
    # 1: use the sLSTM block (not available in this repository)
    _block_map: str = None

    @property
    def block_map(self) -> list[int]:
        return list(map(int, self._block_map.split(",")))

    def _create_block_map(self) -> str:
        """Creates the block map, that specifies which block is used at which position."""
        block_map = [0] * self.num_blocks
        block_map_str = ",".join(map(str, block_map))

        return block_map_str

    def __post_init__(self):
        
        if self.mlstm_block is not None:

            self.mlstm_block.mlstm.embedding_dim = self.embedding_dim
            self.mlstm_block.mlstm.bias = self.bias
            self.mlstm_block.mlstm.dropout = self.dropout
            self.mlstm_block.mlstm.context_length = self.context_length
            self.mlstm_block.mlstm._num_blocks = self.num_blocks
            # call post init, for setting inner_embedding_dim
            self.mlstm_block.__post_init__()

        self._block_map = self._create_block_map()


class xLSTMBlockStack(nn.Module):
    config_class = xLSTMBlockStackConfig

    def __init__(self, config: xLSTMBlockStackConfig):
        super().__init__()
        self.config = config

        self.blocks = self._create_blocks(config=config)
        if config.add_post_blocks_norm:
            self.post_blocks_norm = LayerNorm(ndim=config.embedding_dim)
        else:
            self.post_blocks_norm = nn.Identity()

    def _create_blocks(self, config: xLSTMBlockStackConfig):

        blocks = []
        for block_idx, block_type_int in enumerate(config.block_map):
            if block_type_int == 0:
                config = deepcopy(self.config.mlstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                blocks.append(mLSTMBlock(config=config))
            else:
                raise ValueError(f"Invalid block type {block_type_int}")

        return nn.ModuleList(blocks)

    def reset_parameters(self) -> None:
        for block in self.blocks:
            block.reset_parameters()
        if not isinstance(self.post_blocks_norm, nn.Identity):
            self.post_blocks_norm.reset_parameters()

    def forward(self, x: torch.Tensor, state=None, **kwargs) -> torch.Tensor:

        if self.config.mlstm_block.mlstm.backend not in ["chunkwise", "chunkwise_variable"]:
            state=None

        new_state = {}

        for block_idx, block in enumerate(self.blocks):
            if state != None:
                block_state = state[f"block_{block_idx}"]
            else: 
                block_state = None

            if self.config.mlstm_block.mlstm.return_last_state:

                if self.config.checkpoint_blocks:
                    x, new_state[f"block_{block_idx}"] = checkpoint(block, x, state=block_state, use_reentrant=False, **kwargs)
                else:
                    x, new_state[f"block_{block_idx}"] = block(x, state=block_state, **kwargs)

            else:

                if self.config.checkpoint_blocks:
                    x = checkpoint(block, x, state=block_state, **kwargs, use_reentrant=False)
                else:  
                    x = block(x, state=block_state, **kwargs)

        x = self.post_blocks_norm(x)

        if self.config.mlstm_block.mlstm.return_last_state:
            return x, new_state
        else:
            return x

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None,
        **kwargs
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        if state is None:
            state = {}

        for block_idx, block in enumerate(self.blocks):
            x, state[f"block_{block_idx}"] = block.step(
                x, **state.get(f"block_{block_idx}", {}), **kwargs
            )

        x = self.post_blocks_norm(x)

        return x, state
