__all__ = [
    "xLSTMConfig", 
    "xLSTMLMHeadModel",
]

import json
import os
from collections import namedtuple
from dataclasses import asdict

import torch
import torch.nn as nn
from dacite import Config as DaciteConfig, from_dict
from omegaconf import OmegaConf
from transformers import PretrainedConfig

from protxlstm.generation import GenerationMixinSafe
from protxlstm.utils import load_config_hf, load_state_dict_hf
from protxlstm.xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig


class xLSTMConfig(PretrainedConfig):

    def __init__(self):
        self.config_dataclass = xLSTMLMModelConfig()

    def init_from_dict(self, config: dict):
        config = OmegaConf.create(config)
        self.config_dataclass = from_dict(
            data_class=xLSTMLMModelConfig,
            data=OmegaConf.to_container(config),
            config=DaciteConfig(strict=True),
        )
        return self

    def to_dict(self):
        return asdict(self.config_dataclass)
    

class xLSTMLMHeadModel(nn.Module, GenerationMixinSafe):

    def __init__(self, config: xLSTMConfig) -> None:
        super().__init__()

        self.config = config
        self.backbone = xLSTMLMModel(self.config.config_dataclass)
        self.backbone.reset_parameters()

        self.setup()


    def setup(self):
        
        if 'LOCAL_RANK' in os.environ:
            current_device = int(os.environ['LOCAL_RANK'])
        else:
            if 'SLURM_LOCALID' in os.environ:
                current_device = int(os.environ['SLURM_LOCALID'])
            else:
                current_device = 0

        torch.cuda.set_device(f'cuda:{current_device}')

        self.backbone = self.backbone.to("cuda")


    def forward(
        self,
        input_ids,
        state=None,
        position_ids=None,
        seq_position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        save_layer=[],
        **kwargs,
    ):

        if self.config.config_dataclass.mlstm_block.mlstm.return_last_state:
            lm_logits, state = self.backbone(input_ids, position_ids=position_ids, seq_position_ids=seq_position_ids, state=state)
            CausalLMOutput = namedtuple("CausalLMOutput", ["loss", "logits", "state"])
            return CausalLMOutput(loss=None, logits=lm_logits, state=state)
        else: 
            lm_logits = self.backbone(input_ids, position_ids=position_ids, seq_position_ids=seq_position_ids, state=state)
            CausalLMOutput = namedtuple("CausalLMOutput", ["loss", "logits"])
            return CausalLMOutput(loss=None, logits=lm_logits)
    
    def step(
        self,
        input_ids,
        state=None,
        position_ids=None,
        seq_position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        save_layer=[],
        **kwargs,
    ):

        lm_logits, state = self.backbone.step(
            input_ids, state=state, position_ids=position_ids, seq_position_ids=seq_position_ids
        )

        return lm_logits, state
    

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name,
        device=None,
        dtype=None,
        mlstm_backend=None,
        mlstm_chunksize=None,
        checkpoint_blocks=None,
        rope_base_frequency=None,
        mlstm_return_last_state=None,
    ):
        # Load the checkpoint config
        config_dict = load_config_hf(pretrained_model_name)

        # update rope base frequency
        if rope_base_frequency is not None and config_dict.get("rope_base_frequency", None) != rope_base_frequency:
            config_dict["rope_base_frequency"] = rope_base_frequency
        
        # update mlstm backend
        if mlstm_backend is not None and config_dict["mlstm_block"]["mlstm"].get("backend", None) != mlstm_backend:
            assert mlstm_backend in ["chunkwise", "chunkwise_variable", "parallel"], "invalid mlstm backend."
            config_dict["mlstm_block"]["mlstm"]["backend"] = mlstm_backend

        # update mlstm chunksize
        if mlstm_chunksize is not None and config_dict["mlstm_block"]["mlstm"].get("chunk_size", None) != mlstm_chunksize:
            config_dict["mlstm_block"]["mlstm"]["chunk_size"] = mlstm_chunksize

        # update activation checkpointing
        if checkpoint_blocks is not None:
            config_dict["checkpoint_blocks"] = checkpoint_blocks

        if mlstm_return_last_state is not None:
            config_dict["mlstm_block"]["mlstm"]["return_last_state"] = mlstm_return_last_state
            
        if "slstm_block" in config_dict:
            config_dict.pop("slstm_block")

        if "slstm_at" in config_dict:
            config_dict.pop("slstm_at")

        config = xLSTMConfig().init_from_dict(config_dict)
        
        model = cls(config)

        state_dict = load_state_dict_hf(
            pretrained_model_name, device=device, dtype=dtype
        )
        assert (
            state_dict.keys() == model.state_dict().keys()
        ), "The keys of the state_dict do not match the model's keys."

        model.load_state_dict(state_dict)
        
        return model

    def save_pretrained(self, save_directory):
        """
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
            json.dump(self.config.to_dict(), f)


