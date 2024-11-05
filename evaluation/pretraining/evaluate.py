import os
import shutil
import numpy as np
import pandas as pd

import argparse
from omegaconf import OmegaConf
from transformers import TrainingArguments

from protxlstm.dataloaders import ProteinMemmapDataset, ProteinDataCollator
from protxlstm.models.mamba import MambaLMHeadModelwithPosids
from protxlstm.models.xlstm import xLSTMLMHeadModel
from protxlstm.trainer import ProtTrainer
from protxlstm.utils import AA_TO_ID, compute_metrics, compute_metrics_with_std, load_model

def evaluate(
        model_class,
        model_path,
        max_msa_len=131072,
        subset="test",
        stats=True,
):
    
    """
    Run training loop.
   
    Args:
       config (dict): dictionary with the configuration parameters.     
    """ 

    os.environ["WANDB_MODE"] = "disabled"

    # Get config
    if 'xlstm' in model_class.__name__.lower():
        config_path = 'configs/xlstm_default_config.yaml'
    elif 'mamba' in model_class.__name__.lower():
        config_path = 'configs/mamba_default_config.yaml'
        from protxlstm.models.mamba import MambaLMHeadModelwithPosids
    elif 'transformer' in model_class.__name__.lower():
        config_path = 'configs/llama_default_config.yaml'
    
    model_config = OmegaConf.load(config_path)
    train_config = OmegaConf.load('configs/train_default_config.yaml')
    config = OmegaConf.merge(model_config, train_config)


    # Load dataset
    if subset == "test":
        subset_path = config["test_set"]
    elif subset == "valid":
        subset_path = config["valid_set"]
    elif subset == "train":
        subset_path = config["train_eval_set"]
    else:
        raise ValueError(f"`subset` must be in ['test', 'valid', 'train']")

    dataset = ProteinMemmapDataset(
        msa_memmap_path= 'data/open_protein_set_memmap.dat',
        msa_memmap_meta_path= 'data/open_protein_set_memmap_indices.csv',
        subset_path=subset_path,
        fim_strategy=config["fim_strategy"],
        always_mask=config["always_mask"],
        max_msa_len=max_msa_len,
    )    
    assert (
        len(AA_TO_ID) == config["model"]["vocab_size"]
    ), f"Vocab size in the config file does not match the one in the code. I should be {len(AA_TO_ID)}"    

    data_collator = ProteinDataCollator(max_sequence_length=config["max_msa_len"])
    
    # these fields are updated in the config loaded from the checkpoint
    if model_class == xLSTMLMHeadModel:
        config_update_kwargs = {
            "mlstm_backend": "chunkwise_variable",
            "mlstm_chunksize": 2048,
        }
    else:
        config_update_kwargs = {}

    # Load model
    model = load_model(
        'checkpoints/' + model_path,
        device="cuda:0",
        model_class=model_class,
        **config_update_kwargs,
    )

    # Create model trainer
    trainer = ProtTrainer(
        model=model,
        args=TrainingArguments(
            "tmp",
            local_rank=int(os.getenv('LOCAL_RANK', '0')),
            per_device_eval_batch_size=1,
            bf16=True,
            dataloader_num_workers=32,
            report_to=None,
            label_names=["labels"],
        ),
        compute_only_fim_loss=False,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_std if stats else compute_metrics,
    )

    results = trainer.evaluate(eval_dataset=dataset)

    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process dataset.")

    parser.add_argument("--model_name", type=str, default="protxlstm_102M_60B", help="The name of the checkpoint folder.")
    parser.add_argument("--model_type", type=str, default="xlstm", help="xlstm or mamba.")
    parser.add_argument("--context_len", type=int, default=131072, help="The context length to evaluate on.")

    args = parser.parse_args()

    model_name = args.model_name
    model_type = args.model_type
    context_len = args.context_len

    if model_type =="xlstm":
        model_class = xLSTMLMHeadModel
    elif model_type == "mamba":
        model_class = MambaLMHeadModelwithPosids
    else:
        raise f"{model_type} is not a valid model type."

    subsets = ['test', 'valid', 'train']
    stats = True
    file_path = f'evaluation/pretraining/metrics_{model_name}.csv' if stats else f'evaluation/pretraining/metrics_without_statistics_{model_name}.csv'

    for subset in subsets:

        results = evaluate(model_class = model_class, model_path = model_name, max_msa_len=context_len, subset=subset, stats=stats)
        
        if stats:
            sample_size = 500 if subset == 'test' else 192
            ci_factor = 1.96 / np.sqrt(sample_size)
            results.update({col[:-3] + 'ci95': results[col] * ci_factor for col in results if col.endswith('std')})
            results = dict(sorted(results.items()))
        
        df = pd.DataFrame({**{'model': model_name, 'subset': subset}, **results}, index=[0])

        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df_existing = pd.read_csv(file_path)
            df_combined = pd.concat([df_existing, df], ignore_index=True)
            df_combined.to_csv(file_path, index=False)         

        shutil.rmtree('tmp')   
   