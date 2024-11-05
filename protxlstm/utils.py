# Some of the objects in this file come from ProtMamba and mamba both under Apache License 2.0.

import json
import os

import numpy as np
import rich
import torch
from Bio import SeqIO
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
import wandb
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

__all__ = ['AA_TO_ID', 'MASK_TO_ID', 'ID_TO_AA', 'load_model', 'encode_sequence', 'decode_sequence', 'clean_sequence', 'tokenizer',
           'reorder_masked_sequence', 'load_sequences_from_msa_file', 'prepare_dataset_for_fim_generation',
           'prepare_tokens', 'prepare_target', 'print_number_of_parameters', 'find_fim_indices',
           'compute_metrics', 'compute_metrics_with_std', 'print_config', 'print_zero_rank', 'is_zero_rank']

# Constants
AA_TO_ID = {'<cls>': 0,
            '<pad>': 1,
            '<eos>': 2,
            '<unk>': 3,
            'L': 4,
            'A': 5,
            'G': 6,
            'V': 7,
            'S': 8,
            'E': 9,
            'R': 10,
            'T': 11,
            'I': 12,
            'D': 13,
            'P': 14,
            'K': 15,
            'Q': 16,
            'N': 17,
            'F': 18,
            'Y': 19,
            'M': 20,
            'H': 21,
            'W': 22,
            'C': 23,
            'X': 24,
            'B': 25,
            'U': 26,
            'Z': 27,
            'O': 28,
            '.': 29,
            '-': 30,
            '<null_1>': 31,
            '<mask>': 32}

MASK_TO_ID = {"<mask-1>": 33,
              "<mask-2>": 34,
              "<mask-3>": 35,
              "<mask-4>": 36,
              "<mask-5>": 37,}

AA_TO_ID.update(MASK_TO_ID)

ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}

# Logging & prints
def setup_wandb(config):

    # WandB setup
    os.environ["WANDB_PROJECT"] = config["wandb_project"]
    os.environ["WANDB_ENTITY"] = config["wandb_entity"]
    os.environ["WANDB_MODE"] = config["wandb_mode"]

    if config['model_type'] == 'xlstm':
        pe = config['model']['add_position_ids']
        pe  = 'None' if pe == 'none' else 'AbsPE' if pe == 'abs_1d' else 'AbsPE2' if pe == 'abs_2d' else 'RoPE' if pe == 'rot_1d' else pe == 'rot_2d'
        wandb_run_name = f"{config['model_type']}_l{config['model']['num_blocks']}_d{config['model']['embedding_dim']}_{pe}_s{config['max_msa_len']}_lr{config['learning_rate']}"
    elif config['model_type'] == 'mamba':
        pe = config['model']['add_position_ids']
        pe = 'None' if pe == 'none' else 'AbsPE' if pe == '1d' else pe == '2d'
        wandb_run_name = f"{config['model_type']}_l{config['model']['n_layer']}_d{config['model']['d_model']}_{pe}_s{config['max_msa_len']}_lr{config['learning_rate']}"
    elif config['model_type'] == 'llama':
        pe = 'RoPE'
        wandb_run_name = f"{config['model_type']}_l{config['model']['n_layer']}_d{config['model']['d_model']}_dh{config['model']['hidden_dim']}_{prepare_dataset_for_fim_generation}_s{config['max_msa_len']}_lr{config['learning_rate']}_sched-{config['scheduler']}"

    if config['name_prefix']:
        wandb_run_name = str(config['name_prefix']) + '_' + wandb_run_name
    if config['name_suffix']:
        wandb_run_name = wandb_run_name + '_' + str(config['name_suffix'])

    if is_zero_rank():
        wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            mode=config["wandb_mode"],
            name=wandb_run_name)
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb.config.update(config_dict)
    return wandb_run_name

def is_zero_rank():
    return int(os.getenv('LOCAL_RANK', '0')) == 0

def print_zero_rank(var):
    if is_zero_rank():
        print(var)

def print_number_of_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    formatted_num_params = f"{num_params:_}"
    print("Number of trainable parameters: ", formatted_num_params)

# Sequence tools
def encode_sequence(sequence):
    """Tokenize a sequence of amino acids and add a cls token at the beginning."""
    tokenized_sequence = [AA_TO_ID[aa] if aa in AA_TO_ID else AA_TO_ID['<unk>'] for aa in sequence]
    return [AA_TO_ID['<cls>']] + tokenized_sequence

def decode_sequence(sequence):
    """Decode a sequence of tokens."""
    return "".join([ID_TO_AA[token] if token in ID_TO_AA else "<unk>" for token in sequence])

def clean_sequence(sequence):
    """Remove gaps and convert all residues to upper case."""
    return sequence.replace("-", "").upper()

def tokenizer(sequence_list, concatenate=True):
    """Tokenize a collection of sequences. If the sequences are aligned, the gaps will be removed
    and the insertions (lower case) will be promoted to upper case."""
    # clean and encode all sequences
    sequence_list = [encode_sequence(clean_sequence(sequence)) for sequence in sequence_list]
    if concatenate:
        # concatenate all sequences
        sequences = np.concatenate(sequence_list)
        # convert to tensor and add batch dimension
        return torch.asarray(sequences, dtype=torch.int8)[None,:]
    else:
        return [torch.asarray(sequence, dtype=torch.int8) for sequence in sequence_list]

def reorder_masked_sequence(mask_seq, return_ids=False):
    """
    Reorder a masked sequence to fill the masked positions with the tokens
    that should be there but are positioned after the <eos> token.
    """
    mask_seq = mask_seq.split("<cls>")[0]
    try:
        # Split the sequence and masks
        seq, masks = mask_seq.split("<eos>")
    except:
        return mask_seq
    full_seq = ""
    ids_mask = []
    # Iterate over each mask tag
    for mm in ["<mask-1>", "<mask-2>", "<mask-3>", "<mask-4>", "<mask-5>","<mask-?>"]:
        try:
            # Split the sequence in before and after the mask tag
            seq1, seq2 = seq.split(mm)
            if mm=="<mask-1>":
                # If the mask is the first one, add the sequence before the mask and update the masks
                masks = masks.split("<mask-1>")[1]
                full_seq += seq1
            else:
                # If the mask is not the first one, insert the mask between the two sequence parts
                masks1, masks2 = masks.split(mm)
                ids_mask += [(len(full_seq), len(full_seq)+len(masks1))]
                full_seq += masks1 + seq1
                # Update the masks
                masks = masks2 
            # Update the sequence with the part after the mask
            seq = seq2
        except:
            # If the mask is not found, add the remaining sequence
            ids_mask += [(len(full_seq), len(full_seq)+len(masks))]
            full_seq += masks + seq
            break
    if return_ids:
        return full_seq, ids_mask
    return full_seq

def load_sequences_from_msa_file(file_path):
    """Load a collection of sequences from an a3m file."""
    with open(file_path, "r") as f:
        sequences = [str(record.seq) for record in SeqIO.parse(f, "fasta")]
    return sequences

def prepare_dataset_for_fim_generation(tokens, pos_ids):
    """
    Function to transform the tokenized training dataset into a format that can be used for FIM generation.
    Splits the input tokens and pos_ids into the FIM part (of the last sequence) and the context part (all
    the previous sequences and the masked part of the last sequence).
    Also returns a dictionary with the positions of the mask tokens in the FIM part.
    """
    def find_mask_positions(tokens_fim):
        """
        Function to find the positions of the mask tokens in the FIM part of the last sequence.
        """
        bool_mask = None
        inds_masks = []
        for ind in MASK_TO_ID.values():
            tmp_bool = tokens_fim[0].cpu().numpy() == ind
            bool_mask = tmp_bool if bool_mask is None else bool_mask | tmp_bool
            inds_masks += [ind]
        return bool_mask, inds_masks
    # find where the FIM part of the last sequence starts
    start_last_fim = np.where(tokens[0].cpu().numpy() == AA_TO_ID["<eos>"])[0][-1]
    start_next_seqs = np.where(tokens[0,start_last_fim+1:].cpu().numpy() == AA_TO_ID["<cls>"])[0]
    end_last_fim = start_last_fim+ 1 +start_next_seqs[0] if len(start_next_seqs) > 0 else tokens.shape[1]
    # split tokens and pos_ids into FIM part and context part
    tokens_to_fim = tokens[:,:start_last_fim+1]
    pos_ids_to_fim = pos_ids[:,:start_last_fim+1]
    tokens_fim = tokens[:,start_last_fim+1:end_last_fim]
    pos_ids_fim = pos_ids[:,start_last_fim+1:end_last_fim]
    # find positions of mask tokens
    bool_mask, inds_masks = find_mask_positions(tokens_fim)
    masked_positions = pos_ids_fim[0,bool_mask]
    mask_dict = {ind: int(pos) for ind, pos in zip(inds_masks, masked_positions)}
    return tokens_to_fim, pos_ids_to_fim, tokens_fim, pos_ids_fim, mask_dict

# Metrics
def find_fim_indices(is_cls_tokens, is_eos_tokens):
    """Function to find the indices of the FIM tokens in the sequences.
    """
    # add a cls token at the beginning
    is_cls_tokens = torch.cat([torch.ones_like(is_cls_tokens[:, :1]), is_cls_tokens], dim=1)
    is_eos_tokens = torch.cat([torch.zeros_like(is_eos_tokens[:, :1]), is_eos_tokens], dim=1)
    # both eos and cls tokens
    bol = is_cls_tokens | is_eos_tokens
    tmp = torch.zeros_like(is_cls_tokens, dtype=torch.int)
    tmp[torch.nonzero(is_cls_tokens, as_tuple=True)] = 1
    tmp[torch.nonzero(is_eos_tokens, as_tuple=True)] = -1
    bol1 = torch.clone(bol)
    for batch_ind in range(tmp.size(0)):
        tmp1 = tmp[batch_ind,bol[batch_ind]]
        # find all positions where a 1 if preceeded by a -1
        tmp1 = tmp1[:-1]*tmp1[1:]
        # add the first element to make the sequence start with a 1
        tmp1 = torch.cat([torch.ones_like(tmp1[:1]).to(tmp1.device), tmp1])
        new_bol = tmp1<0
        # bool array True only in the positions where a 1 is preceeded by a -1
        bol1[batch_ind,bol[batch_ind]] = False if new_bol.size(0) == 0 else new_bol
    cumulative_sum = torch.cumsum(bol1, dim=1)
    # Use modulo operation to get the desired tensor
    bol2 = cumulative_sum % 2 == 1
    bol2[is_eos_tokens]= False
    return bol2[:,1:]

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions).permute(0, 2, 1)
    labels = torch.tensor(labels)
    # shift labels to align them with predictions and remove last prediction to match the length
    predictions = predictions[:, :, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    # compute unreduced elementwise loss
    unreduced_loss = torch.nn.functional.cross_entropy(predictions, labels, reduction="none")
    # compute reconstruction accuracy
    reconstruction = (predictions.argmax(1) == labels)

    # start and end tokens
    is_cls_tokens = (labels == AA_TO_ID["<cls>"])
    is_eos_tokens = (labels == AA_TO_ID["<eos>"])
    # fill in the middle tokens
    if False:
        fim_tokens = torch.zeros(is_cls_tokens.size(0), is_cls_tokens.size(1), dtype=torch.bool)
        in_mask_vector = torch.zeros(is_cls_tokens.size(0), dtype=torch.bool)
        for j in range(is_cls_tokens.size(1)):
            in_mask_vector = in_mask_vector & ~is_cls_tokens[:, j]
            fim_tokens[:, j] = in_mask_vector
            in_mask_vector = in_mask_vector | is_eos_tokens[:, j]
    fim_tokens = find_fim_indices(is_cls_tokens, is_eos_tokens)
        
    number_sequences = torch.cumsum(torch.cat([torch.zeros(is_cls_tokens.size(0),1, dtype=torch.int32), is_cls_tokens[:,:-1]],1), -1)
    # fist, second and last sequence tokens
    first_sequence_tokens = ((~fim_tokens & (labels < 33)) | fim_tokens) & (number_sequences == 0)
    second_sequence_tokens = ((~fim_tokens & (labels < 33)) | fim_tokens) & (number_sequences == 1)
    last_sequence_tokens = ((~fim_tokens & (labels < 33)) | fim_tokens) & (number_sequences == (number_sequences.max(1).values[:, None] - 1))
    # end of mask tokens
    end_of_masks = (fim_tokens & (labels > 33)) | is_cls_tokens | is_eos_tokens

    return {
        "loss/all": torch.mean(unreduced_loss).item(),
        "loss/end_span": torch.mean(unreduced_loss[end_of_masks]).item(),
        "perplexity/seq": torch.mean(torch.exp(torch.mean(unreduced_loss, dim=1))).item(),
        "perplexity/end_span": torch.exp(torch.mean(unreduced_loss[end_of_masks])).item(),
        "perplexity/batch": torch.exp(torch.mean(unreduced_loss)).item(),
        "perplexity/first_seq": torch.exp(torch.mean(unreduced_loss[first_sequence_tokens])).item(),
        "perplexity/second_seq": torch.exp(torch.mean(unreduced_loss[second_sequence_tokens])).item(),
        "perplexity/last_seq": torch.exp(torch.mean(unreduced_loss[last_sequence_tokens])).item(),
        "perplexity/fim": torch.exp(torch.mean(unreduced_loss[fim_tokens])).item(),
        "reconstruction/all": torch.mean(reconstruction.float()).item(),
        "reconstruction/end_span": torch.mean(reconstruction[end_of_masks].float()).item(),
        "reconstruction/first_seq": torch.mean(reconstruction[first_sequence_tokens].float()).item(),
        "reconstruction/second_seq": torch.mean(reconstruction[second_sequence_tokens].float()).item(),
        "reconstruction/last_seq": torch.mean(reconstruction[last_sequence_tokens].float()).item(),
        "reconstruction/fim": torch.mean(reconstruction[fim_tokens].float()).item(),
        }

def compute_metrics_with_std(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions).permute(0, 2, 1)
    labels = torch.tensor(labels)
    # shift labels to align them with predictions and remove last prediction to match the length
    predictions = predictions[:, :, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    # compute unreduced elementwise loss
    unreduced_loss = torch.nn.functional.cross_entropy(predictions, labels, reduction="none")
    # compute reconstruction accuracy
    reconstruction = (predictions.argmax(1) == labels)

    # start and end tokens
    is_cls_tokens = (labels == AA_TO_ID["<cls>"])
    is_eos_tokens = (labels == AA_TO_ID["<eos>"])
    # fill in the middle tokens
    if False:
        fim_tokens = torch.zeros(is_cls_tokens.size(0), is_cls_tokens.size(1), dtype=torch.bool)
        in_mask_vector = torch.zeros(is_cls_tokens.size(0), dtype=torch.bool)
        for j in range(is_cls_tokens.size(1)):
            in_mask_vector = in_mask_vector & ~is_cls_tokens[:, j]
            fim_tokens[:, j] = in_mask_vector
            in_mask_vector = in_mask_vector | is_eos_tokens[:, j]
    fim_tokens = find_fim_indices(is_cls_tokens, is_eos_tokens)
        
    number_sequences = torch.cumsum(torch.cat([torch.zeros(is_cls_tokens.size(0),1, dtype=torch.int32), is_cls_tokens[:,:-1]],1), -1)
    # fist, second and last sequence tokens
    first_sequence_tokens = ((~fim_tokens & (labels < 33)) | fim_tokens) & (number_sequences == 0)
    second_sequence_tokens = ((~fim_tokens & (labels < 33)) | fim_tokens) & (number_sequences == 1)
    last_sequence_tokens = ((~fim_tokens & (labels < 33)) | fim_tokens) & (number_sequences == (number_sequences.max(1).values[:, None] - 1))
    # end of mask tokens
    end_of_masks = (fim_tokens & (labels > 33)) | is_cls_tokens | is_eos_tokens
    
    def perplexities_per_seq_for_subset(unreduced_loss, subset):
        return torch.exp(torch.nanmean(torch.where(subset, unreduced_loss, torch.tensor(float('nan'))), dim=1))
    
    return{
        # Loss
        "loss/all": torch.mean(unreduced_loss).item(),
        "loss/std": torch.std(unreduced_loss).item(), 
        "loss/end_span": torch.mean(unreduced_loss[end_of_masks]).item(),
        "loss/end_span_std": torch.std(unreduced_loss[end_of_masks]).item(),

        # Perplexity of all tokens
        "perplexity/batch": torch.exp(torch.mean(unreduced_loss)).item(),
        "perplexity/batch_std": torch.exp(torch.std(unreduced_loss)).item(), # Fix
        
        # Perplexity per sequence
        "perplexity/seq": torch.mean(torch.exp(torch.mean(unreduced_loss, dim=1))).item(),
        "perplexity/seq_std": torch.std(torch.exp(torch.mean(unreduced_loss, dim=1))).item(),
        "perplexity/end_span": torch.exp(torch.mean(unreduced_loss[end_of_masks])).item(),
        "perplexity/end_span_std": torch.std(torch.exp(unreduced_loss[end_of_masks])).item(),   
        
        "perplexity/first_seq": torch.mean(perplexities_per_seq_for_subset(unreduced_loss, first_sequence_tokens)).item(),
        "perplexity/first_seq_std": torch.std(perplexities_per_seq_for_subset(unreduced_loss, first_sequence_tokens)).item(),
        "perplexity/second_seq": torch.mean(perplexities_per_seq_for_subset(unreduced_loss, second_sequence_tokens)).item(),
        "perplexity/second_seq_std": torch.std(perplexities_per_seq_for_subset(unreduced_loss, second_sequence_tokens)).item(),
        "perplexity/last_seq": torch.mean(perplexities_per_seq_for_subset(unreduced_loss, last_sequence_tokens)).item(),
        "perplexity/last_seq_std": torch.std(perplexities_per_seq_for_subset(unreduced_loss, last_sequence_tokens)).item(),
        "perplexity/fim": torch.mean(perplexities_per_seq_for_subset(unreduced_loss, fim_tokens)).item(),
        "perplexity/fim_std": torch.std(perplexities_per_seq_for_subset(unreduced_loss, fim_tokens)).item(),
        "reconstruction/all": torch.mean(reconstruction.float()).item(),
        "reconstruction/std": torch.std(reconstruction.float()).item(),
        "reconstruction/end_span": torch.mean(reconstruction[end_of_masks].float()).item(),
        "reconstruction/end_span_std": torch.std(reconstruction[end_of_masks].float()).item(),
        "reconstruction/first_seq": torch.mean(reconstruction[first_sequence_tokens].float()).item(),
        "reconstruction/first_seq_std": torch.std(reconstruction[first_sequence_tokens].float()).item(),
        "reconstruction/second_seq": torch.mean(reconstruction[second_sequence_tokens].float()).item(),
        "reconstruction/second_seq_std": torch.std(reconstruction[second_sequence_tokens].float()).item(),
        "reconstruction/last_seq": torch.mean(reconstruction[last_sequence_tokens].float()).item(),
        "reconstruction/last_seq_std": torch.std(reconstruction[last_sequence_tokens].float()).item(),
        "reconstruction/fim": torch.mean(reconstruction[fim_tokens].float()).item(),
        "reconstruction/fim_std": torch.std(reconstruction[fim_tokens].float()).item(),
    }

# Others
def set_optimizer_and_scheduler(config, ntrain, parameters):

    # Set optimizer
    optimizer = AdamW(
        parameters,
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )

    eff_batch_size = config["batch_size"] * config["gradient_accumulation_steps"] * torch.cuda.device_count()

    # Set scheduler
    if config["scheduler"] == "cosine":
        print_zero_rank("Using cosine scheduler")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=config["num_epochs"] * ntrain // eff_batch_size,
        )
    if config["scheduler"] == "cosine-restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=config["num_epochs"] * ntrain // eff_batch_size,
            num_cycles=config["num_cycles"],
        )
    elif config["scheduler"] == "constant":
        print_zero_rank("Using constant scheduler with warmup")
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=config["warmup_steps"]
        )
    else:
        raise ValueError("Scheduler must be either cosine or constant") 
    
    # Finetuning and no optimizer/scheduler reset
    if config.finetune_model_path and not config.restart_optimizer_and_scheduler:
        optimizer.load_state_dict(torch.load(config.finetune_model_path + "/optimizer.pt"))
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = config['learning_rate']
            param_group['lr'] = config['learning_rate']

        scheduler.load_state_dict(torch.load(config.finetune_model_path + "/scheduler.pt"))
        scheduler.base_lrs = [config['learning_rate']]
        scheduler._last_lr = [config['learning_rate']]  

    return optimizer, scheduler 

def parse_override_args(override_args):
    overrides = {}
    for arg in override_args:
        key, value = arg.split("=")
        keys = key.split(".")
        sub_dict = overrides
        for sub_key in keys[:-1]:
            if sub_key not in sub_dict:
                sub_dict[sub_key] = {}
            sub_dict = sub_dict[sub_key]
        # Convert value to appropriate type
        if value == 'True':
            value = True
        elif value == 'False':
            value = False
        elif value == 'None':
            value = None
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        sub_dict[keys[-1]] = value
    return overrides

def load_model(
    model_path,
    device,
    model_class,
    dtype=torch.bfloat16,
    **kwargs
):
    model = model_class.from_pretrained(
        model_path, device=device, dtype=dtype, **kwargs
    )
    return model

# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/hf.py
def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))

# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/hf.py
def load_state_dict_hf(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, map_location=mapped_device)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict