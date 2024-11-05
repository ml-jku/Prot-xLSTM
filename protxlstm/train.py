# Original code from ProtMamba under Apache License 2.0.
#
# Modifications made by Niklas Schmidinger, Lisa Schneckenreiter and Sohvi Luukkonen 
#   - Extended to training of xlstm and transformer-based models
#   - Predefined splits instead of on-the-fly creation
#   - Option to overwrite config parameters from the command line
#   - wandb logging

import argparse
import os

import torch
from omegaconf import OmegaConf
from transformers import TrainingArguments

from protxlstm.dataloaders import ProteinMemmapDataset, ProteinDataCollator
from protxlstm.models.xlstm import xLSTMConfig, xLSTMLMHeadModel
from protxlstm.models.llama import TransformerConfig, TransformerLMHeadModel
from protxlstm.trainer import ProtTrainer, EarlyStoppingCallback, get_last_checkpoint
from protxlstm.utils import (
    AA_TO_ID,
    compute_metrics,
    is_zero_rank,
    parse_override_args,
    print_number_of_parameters,
    print_zero_rank,
    set_optimizer_and_scheduler,
    setup_wandb,
    load_model,
)

def run(config):
    """
    Run training loop.
   
    Args:
       config (dict): dictionary with the configuration parameters.     
    """ 

    if config.model_type == 'llama':
        pe_kwargs = {
            'max_position_embeddings' : config["model"]["max_position_embeddings"],
            'add_position_ids' : '1d',
        }
    elif config.model_type == 'mamba':
        from protxlstm.models.mamba import MambaConfig, MambaLMHeadModelSafe, MambaLMHeadModelwithPosids, MambaLMHeadModelwith2DPosids
        pe_kwargs = {
            'max_position_embeddings' : config["model"]["max_position_embeddings"],
            'max_seq_position_embeddings' : config["model"]["max_seq_position_embeddings"],
            'add_position_ids' : config["model"]["add_position_ids"]
        }
    else:
        position_embeddings = config["model"]["position_embeddings"]
        assert position_embeddings in ["none", "abs_1d", "abs_2d", "rot_1d", "rot_2d"]
        if position_embeddings != "none":
            position_embeddings = position_embeddings.split("_")[-1]
        pe_kwargs = {
            'max_position_embeddings' : config["model"]["max_position_embeddings"],
            'max_seq_position_embeddings' : config["model"]["max_seq_position_embeddings"],
            'add_position_ids' : position_embeddings
        }

    # Setup WandB
    wandb_run_name = setup_wandb(config)

    # Load datasets
    dataset_params = {
        "msa_memmap_path": config["msa_memmap_path"],
        "msa_memmap_meta_path": config["msa_memmap_meta_path"],
        "sample": config["sample_sequences"],
        "max_msa_len": config["max_msa_len"],
        "reverse": False,
        "seed": config["seed_sequence_sampling"],
        "troubleshoot": False,
        "fim_strategy": config["fim_strategy"],
        "always_mask": config["always_mask"],
        **pe_kwargs,
    }
    train_dataset = ProteinMemmapDataset(subset_path=config["train_set"], **dataset_params)
    valid_dataset = ProteinMemmapDataset(subset_path=config["valid_set"], **dataset_params)
    train_eval_dataset = ProteinMemmapDataset(subset_path=config["train_eval_set"], **dataset_params)

    print(f'Train set size: {len(train_dataset)} Train eval set size: {len(train_eval_dataset)} Valid set size: {len(valid_dataset)}')
    
    assert (
        len(AA_TO_ID) == config["model"]["vocab_size"]
    ), f"Vocab size in the config file does not match the one in the code. I should be {len(AA_TO_ID)}"
    
    # Create data collator for batched training
    data_collator = ProteinDataCollator(max_sequence_length=config["max_msa_len"])
 
    # Check datatypes
    if config["dtype"] == "float32":
        dtype = torch.float32
    elif config["dtype"] == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError("dtype must be either float32 or bfloat16")
    
    # Initialize model
    if config.model_type == 'xlstm':
        
        # Load model for finetuning
        if config.finetune_model_path:
            # These fields are updated in the config loaded from the checkpoint
            config_update_kwargs = {
                "mlstm_backend": config["model"]["mlstm_block"]["mlstm"]["backend"],
                "mlstm_chunksize": config["model"]["mlstm_block"]["mlstm"]["chunk_size"],
                "checkpoint_blocks": config["model"]["checkpoint_blocks"],
                "rope_base_frequency": config["model"]["rope_base_frequency"]
            }      
            model = load_model(
                config.finetune_model_path,
                model_class=xLSTMLMHeadModel,
                device="cuda",
                dtype=dtype,
                **config_update_kwargs
            )     
        else:
            # Create new mode
            xlstm_config = xLSTMConfig().init_from_dict(config["model"])
            model = xLSTMLMHeadModel(xlstm_config)                  

    elif config.model_type == 'mamba':

        _mamba_model = {
            "none": MambaLMHeadModelSafe, 
            "1d": MambaLMHeadModelwithPosids, 
            "2d": MambaLMHeadModelwith2DPosids, 
            }
        Mamba = _mamba_model[config['model']["add_position_ids"]]

        # Load model for finetuning
        if config.finetune_model_path:     
            model = load_model(
                config.finetune_model_path,
                model_class=Mamba,
                device="cuda",
                dtype=dtype,
                checkpoint_mixer=config["checkpoint_mixer"],
            )     
        else:
            # Create new mode
            mamba_config = MambaConfig(d_model=config['model']["d_model"],
                                    n_layer=config['model']["n_layer"],
                                    vocab_size=config['model']["vocab_size"],
                                    residual_in_fp32=config['model']["residual_in_fp32"])
            model = Mamba(mamba_config, dtype=dtype, checkpoint_mixer=config['model']["checkpoint_mixer"])
        
    elif config.model_type == 'llama':
        
        llama_config = TransformerConfig(
            d_model=config["model"]["d_model"],
            n_layer=config["model"]["n_layer"],
            n_heads=config["model"]["n_heads"],
            n_kv_heads=config["model"]["n_kv_heads"],
            bidirectional=config["model"]["bidirectional"],
            hidden_dim=config["model"]["hidden_dim"],
            multiple_of=config["model"]["multiple_of"],
            norm_eps=config["model"]["norm_eps"],
            max_length=config["model"]["max_length"],
            vocab_size=config["model"]["vocab_size"],
            dropout=config["model"]["dropout"],
            max_position_embeddings=config["model"]["max_position_embeddings"],
            rope_base_frequency=config["model"]["rope_base_frequency"],

        )

        model = TransformerLMHeadModel(llama_config)

    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}. Expected 'xlstm', 'mamba', or 'llama'.")

    
    # TODO: Improve what we want print
    if is_zero_rank():
        print_number_of_parameters(model)
    print_zero_rank(f"dtype: {config['dtype']}")
    print_zero_rank(f"Epochs: {config['num_epochs']}")
    print_zero_rank(f"Batch size per GPU: {config['batch_size']}")
    print_zero_rank(f"Gradient accumulation steps: {config['gradient_accumulation_steps']}")
    eff_batch_size = config["batch_size"] * config["gradient_accumulation_steps"]
    nr_gpus = torch.cuda.device_count()
    print_zero_rank(f"GPUS: {nr_gpus}")
    eff_batch_size *= nr_gpus
    print_zero_rank(f"Effective batch size: {eff_batch_size}")
    print_zero_rank(
        f"Steps per training epoch: {len(train_dataset) // config['batch_size']}, eff. steps: {len(train_dataset) // eff_batch_size}"
    )
    print_zero_rank(f"Steps per evaluation epoch: {len(valid_dataset) // config['batch_size']}")
    print_zero_rank(f"Max MSA length: {config['max_msa_len']}")
    ev_epochs = round(
        config["eval_steps"] * config["batch_size"] / len(train_dataset), 3
    )
    print_zero_rank(
        f"Evaluation every {config['eval_steps']} steps, i.e. {ev_epochs} epochs. Effectively every {config['eval_steps']*config['gradient_accumulation_steps']} steps, i.e. {ev_epochs*config['gradient_accumulation_steps']} epochs."
    )
    if config.model_type == 'xlstm' and config["model"]["checkpoint_blocks"]:
        print_zero_rank("Using gradient checkpointing")
    if config["compute_only_fim_loss"]:
        print_zero_rank("Computing only FIM loss for training")

    # Training callbacks
    es_callback = EarlyStoppingCallback(
        train_path=config["output_dir"] + '/' + wandb_run_name, config=config
    )
    callbacks = [es_callback]    

    # Optimizer and Schedulers
    optimizer, scheduler = set_optimizer_and_scheduler(
        config, 
        len(train_dataset), 
        model.parameters()
    )

    # Find checkpoint if available
    last_checkpoint = None
    if config.finetune_model_path is None:
        path = os.path.join(config["output_dir"], wandb_run_name)
        if os.path.exists(path):
            last_checkpoint = get_last_checkpoint(path)
            if last_checkpoint is None:
                print_zero_rank("No checkpoint found, starting training from scratch.")
            else:
                print_zero_rank(f"Resuming training from the last checkpoint: {last_checkpoint}")

    # Create trainer
    trainer = ProtTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset={"valid": valid_dataset, "train": train_eval_dataset},
        optimizers=(optimizer, scheduler),
        args=TrainingArguments(
            run_name=wandb_run_name,
            local_rank=int(os.getenv('LOCAL_RANK', '0')),
            learning_rate=config["learning_rate"],
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            eval_accumulation_steps=config["eval_accumulation_steps"],
            eval_strategy="steps",
            max_grad_norm=config["max_grad_norm"],
            bf16=config["dtype"] == "bfloat16",
            dataloader_num_workers=32,
            logging_steps=config["logging_steps"],
            eval_steps=config["eval_steps"],
            save_steps=config["save_steps"],
            output_dir=config["output_dir"] + '/' + wandb_run_name,
            logging_dir=config["output_dir"] + '/' + wandb_run_name,
            report_to="wandb" if is_zero_rank() else None,
            log_on_each_node=False,
            overwrite_output_dir=False,
            push_to_hub=False,
            label_names=["labels"],
        ),
        compute_only_fim_loss=config["compute_only_fim_loss"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )    

    # Train model
    while True:
        if last_checkpoint is None and trainer.state.global_step == 0:
            eval_results = trainer.evaluate()
            print_zero_rank(
                f">>> Initial validation perplexity: {eval_results['eval_valid_perplexity/batch']:.2f}"
            )
        else:
            print_zero_rank(f"Resuming training from the last checkpoint: {last_checkpoint}")
        # Train
        trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # Break training when the number of epochs is reached
        if (
            not es_callback.should_restart
            or trainer.state.epoch >= config["num_epochs"]
        ):
            eval_results = trainer.evaluate()
            print_zero_rank(
                f">>> Final Perplexity: {eval_results['eval_valid_perplexity/batch']:.2f}"
            )
            break
        # If the training was interrupted because of a loss spike, restart from the last checkpoint
        last_checkpoint = es_callback.checkpoint_path

    return trainer

if __name__ == "__main__":

    # Default configuration file paths
    default_model_config = "configs/xlstm_default_config.yaml"
    default_train_config = "configs/train_default_config.yaml"

    parser = argparse.ArgumentParser(
        description="Train or finetune a model with the provided configuration."
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=default_model_config,
        help=f"Path to the model configuration file (default: {default_model_config})"
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        default=default_train_config,
        help=f"Path to the training and dataset configuration file (default: {default_train_config})"
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Override configuration values using key=value format.",
    )

    args = parser.parse_args()  

    # Check if the default config files exist, or raise an error
    if not os.path.exists(args.model_config_path):
        raise FileNotFoundError(f"Model config file not found: {args.model_config_path}")
    if not os.path.exists(args.train_config_path):
        raise FileNotFoundError(f"Train config file not found: {args.train_config_path}")

    # Load the model and training configurations
    model_config = OmegaConf.load(args.model_config_path)
    train_config = OmegaConf.load(args.train_config_path)

    # Merge the model and training configurations
    config = OmegaConf.merge(model_config, train_config)

    # Parse overrides
    if args.overrides:
        overrides = parse_override_args(args.overrides)
        config.merge_with(OmegaConf.create(overrides))

    # Run the training/finetuning process
    run(config)