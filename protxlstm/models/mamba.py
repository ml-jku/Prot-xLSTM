# Original code from ProtMamba under Apache License 2.0.

import json
import os
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Block, Mamba
from mamba_ssm.models.mixer_seq_simple import MixerModel, _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig

from protxlstm.generation import GenerationMixinSafe

@dataclass
class MambaConfig(PretrainedConfig):
    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    max_position_embeddings: int = 2048

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    checkpoint_mixer=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    if checkpoint_mixer:
        block.mixer = CheckpointedModule(block.mixer)
    return block

class CheckpointedModule(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.ckpt_layer = layer

    def forward(self, x, *args, **kwargs):
        return checkpoint(self.ckpt_layer, x, use_reentrant=False)

    # def state_dict(self, **kwargs):
    #     # Get the state dict of the underlying layer
    #     layer_state_dict = self.ckpt_layer.state_dict(**kwargs)
    #     # Create a new state dict with the original keys
    #     state_dict = {k.replace('ckpt_layer.', ''): v for k, v in layer_state_dict.items()}
    #     return state_dict

class MixerModelSafe(MixerModel):
    """
    Overwrite the forward method to allow saving intermediate layers.
    """

    def forward(self, input_ids, inference_params=None, save_layer=[]):
        hidden_states = self.embedding(input_ids)
        residual = None
        if len(save_layer) > 0:
            hidden_states_dict = {}
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if i + 1 in save_layer:
                hidden_states_dict[i + 1] = (
                    hidden_states.detach().cpu().to(torch.float).numpy()
                )
        if len(save_layer) > 0:
            return hidden_states_dict

        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states

class MixerModelWithPosids(nn.Module):
    r"""Mixer model for Mamba but we add positional encodings to the input embeddings."""

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        max_position_embeddings: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        checkpoint_mixer=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model // 2, **factory_kwargs)
        self.position_embedding = nn.Embedding(
            max_position_embeddings, d_model - d_model // 2, **factory_kwargs
        )

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    checkpoint_mixer=checkpoint_mixer,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, position_ids, inference_params=None, save_layer=[]):
        hidden_states = torch.cat(
            [
                self.embedding(input_ids),
                self.position_embedding(position_ids),
            ],
            -1,
        )
        residual = None
        if len(save_layer) > 0:
            hidden_states_dict = {}
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if i + 1 in save_layer:
                hidden_states_dict[i + 1] = (
                    hidden_states.detach().cpu().to(torch.float).numpy()
                )
        if len(save_layer) > 0:
            return hidden_states_dict

        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states

class MixerModelWith2DPosids(nn.Module):
    r"""Mixer model for Mamba but we add positional encodings to the input embeddings."""

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        max_position_embeddings: int,
        max_sequence_position_embeddings: int = 512,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        checkpoint_mixer=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(
            vocab_size, d_model - 2 * d_model // 4, **factory_kwargs
        )
        self.position_embedding = nn.Embedding(
            max_position_embeddings, d_model // 4, **factory_kwargs
        )
        self.seq_position_embedding = nn.Embedding(
            max_sequence_position_embeddings, d_model // 4, **factory_kwargs
        )
        self.d_embeddings = d_model - 2 * d_model // 4

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    checkpoint_mixer=checkpoint_mixer,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids,
        position_ids,
        seq_position_ids,
        inference_params=None,
        save_layer=[],
    ):
        hidden_states = torch.cat(
            [
                self.embedding(input_ids),
                self.position_embedding(position_ids),
                self.seq_position_embedding(seq_position_ids),
            ],
            -1,
        )
        residual = None
        if len(save_layer) > 0:
            hidden_states_dict = {}
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if i + 1 in save_layer:
                hidden_states_dict[i + 1] = (
                    hidden_states.detach().cpu().to(torch.float).numpy()
                )
        if len(save_layer) > 0:
            return hidden_states_dict

        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states

class MambaLMHeadModelSafe(nn.Module, GenerationMixinSafe):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
        checkpoint_mixer=False,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}
        if checkpoint_mixer:
            raise NotImplementedError(
                "Checkpointing is not yet supported for MambaLMHeadModelSafe"
            )

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )
        self.backbone = MixerModelSafe(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def clip_grad_norm_(self, max_norm, norm_type=2.0):
        r"""Clip the norm of the gradients for the model.
        Args:
            max_norm (float or int): The maximum norm of the gradients.
                The gradients are modified in-place.
            norm_type (float or int): The type of the used p-norm. Can be 'inf' for infinity norm.
        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        return torch.nn.utils.clip_grad_value_(self.parameters(), max_norm)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        save_layer=[],
        *args,
        **kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        return self.protected_forward(
            input_ids, position_ids, inference_params, num_last_tokens, save_layer
        )

    def protected_forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        save_layer=[],
    ):
        hidden_states = self.backbone(
            input_ids, inference_params=inference_params, save_layer=save_layer
        )
        if len(save_layer) > 0:
            return hidden_states
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["loss", "logits"])
        return CausalLMOutput(loss=None, logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(
            load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype),
            strict=False,
        )
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f)

class MambaLMHeadModelwithPosids(nn.Module, GenerationMixinSafe):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
        checkpoint_mixer=False,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        max_position_embeddings = config.max_position_embeddings
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )
        self.backbone = MixerModelWithPosids(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            checkpoint_mixer=checkpoint_mixer,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        save_layer=[],
        *args,
        **kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        return self.protected_forward(
            input_ids, position_ids, inference_params, num_last_tokens, save_layer
        )

    def protected_forward(
        self,
        input_ids,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        save_layer=[],
    ):
        hidden_states = self.backbone(
            input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            save_layer=save_layer,
        )
        if len(save_layer) > 0:
            return hidden_states
        hidden_states = hidden_states[:, :, : self.config.d_model // 2]
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["loss", "logits"])
        return CausalLMOutput(loss=None, logits=lm_logits)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name,
        device=None,
        dtype=None,
        checkpoint_mixer=False,
        **kwargs,
    ):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(
            config,
            device=device,
            dtype=dtype,
            checkpoint_mixer=checkpoint_mixer,
            **kwargs,
        )
        state_dict = load_state_dict_hf(
            pretrained_model_name, device=device, dtype=dtype
        )
        if state_dict.keys() != model.state_dict().keys():
            if checkpoint_mixer:
                for key in model.state_dict().keys():
                    if "ckpt_layer" in key:
                        state_dict[key] = state_dict.pop(key.replace("ckpt_layer.", ""))
                print(
                    "Using a model that was pretrained without gradient checkpointing and now want to use it. Changed the keys of the state_dict to match the model's keys."
                )
            else:
                for key in list(state_dict.keys()):
                    if "ckpt_layer" in key:
                        state_dict[key.replace("ckpt_layer.", "")] = state_dict.pop(key)
                print(
                    "Using a model that was pretrained with gradient checkpointing but now do not want to use it. Changed the keys of the state_dict to match the model's keys."
                )
            assert (
                state_dict.keys() == model.state_dict().keys()
            ), "The keys of the state_dict do not match the model's keys."
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f)

class MambaLMHeadModelwith2DPosids(nn.Module, GenerationMixinSafe):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
        checkpoint_mixer=False,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        max_position_embeddings = config.max_position_embeddings
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )
        self.backbone = MixerModelWith2DPosids(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            checkpoint_mixer=checkpoint_mixer,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids,
        position_ids=None,
        seq_position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        save_layer=[],
        *args,
        **kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        return self.protected_forward(
            input_ids,
            position_ids,
            seq_position_ids,
            inference_params,
            num_last_tokens,
            save_layer,
        )

    def protected_forward(
        self,
        input_ids,
        position_ids=None,
        seq_position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        save_layer=[],
    ):
        hidden_states = self.backbone(
            input_ids,
            position_ids=position_ids,
            seq_position_ids=seq_position_ids,
            inference_params=inference_params,
            save_layer=save_layer,
        )
        if len(save_layer) > 0:
            return hidden_states
        hidden_states = hidden_states[:, :, : self.backbone.d_embeddings]
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["loss", "logits"])
        return CausalLMOutput(loss=None, logits=lm_logits)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name,
        device=None,
        dtype=None,
        checkpoint_mixer=False,
        **kwargs,
    ):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(
            config,
            device=device,
            dtype=dtype,
            checkpoint_mixer=checkpoint_mixer,
            **kwargs,
        )
        state_dict = load_state_dict_hf(
            pretrained_model_name, device=device, dtype=dtype
        )
        if state_dict.keys() != model.state_dict().keys():
            if checkpoint_mixer:
                for key in model.state_dict().keys():
                    if "ckpt_layer" in key:
                        state_dict[key] = state_dict.pop(key.replace("ckpt_layer.", ""))
                print(
                    "Using a model that was pretrained without gradient checkpointing and now want to use it. Changed the keys of the state_dict to match the model's keys."
                )
            else:
                for key in list(state_dict.keys()):
                    if "ckpt_layer" in key:
                        state_dict[key.replace("ckpt_layer.", "")] = state_dict.pop(key)
                print(
                    "Using a model that was pretrained with gradient checkpointing but now do not want to use it. Changed the keys of the state_dict to match the model's keys."
                )
            assert (
                state_dict.keys() == model.state_dict().keys()
            ), "The keys of the state_dict do not match the model's keys."
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f)
