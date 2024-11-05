# Original code from ProtMamba under Apache License 2.0.
#
# Modifications made by Niklas Schmidinger, Lisa Schneckenreiter and Sohvi Luukkonen 
#   - Uniclust30_Dataset renamed to ProteinMemmapDataset
#       - Dataset input file format changed for more efficient dataloading
#       - Option to use only a subset
#   - DataCollatorForUniclust30Dataset renamed to ProteinDataCollator
#       - Add sequence padding

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Sequence

from protxlstm.fim import MultipleSpanFIM, NoFIM, SingleSpanFIM
from protxlstm.utils import AA_TO_ID


# Make dataset
class ProteinMemmapDataset(Dataset):
    """
    ProteinMemmapDataset is a PyTorch Dataset class for handling memory-mapped datasets of protein multiple sequence alignments (MSAs).
    
    This class imports MSA data stored in memmap format and associated metadata CSVs. It supports flexible
    data sampling strategies and inpainting methods for sequence manipulation and training purposes.
    
    Args:
        msa_memmap_path (str): Path to the memory-mapped file containing the MSA clusters.
        msa_memmap_meta_path (str): Path to the CSV file with metadata linking MSA Cluster IDs and indices in the memmap array.
        subset_path (str, optional): Path to a CSV file specifying a subset of cluster IDs to use.
        sample (bool, optional): If True, randomly samples sequences from each cluster; otherwise, loads all sequences and shuffles them.
        max_msa_len (int, optional): Maximum length of the MSA sequences to include. Defaults to -1 (no limit).
        reverse (bool, optional): If True, reverses sequences with a probability of 0.5 and moves the last token to the front.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        troubleshoot (bool, optional): If True, prints debugging information. Defaults to False.
        fim_strategy (str, optional): Strategy for inpainting ("no-scramble", "one_span", or "multiple_span").
        max_patches (int, optional): Number of patches for inpainting. Used when fim_strategy is "multiple_span".
        mask_fraction (float, optional): Fraction of the patches to mask. Used when fim_strategy is "multiple_span".
        always_mask (bool, optional): If True, ensures masking is applied in the inpainting process.
        max_position_embeddings (int, optional): Maximum position embeddings. Defaults to 2048.
        max_seq_position_embeddings (int, optional): Maximum sequence position embeddings for 2D positional IDs. Defaults to 512.
        add_position_ids (str, optional): Type of position IDs to add ("none", "1d", or "2d"). Defaults to "1d".
    """
    
    _FIM = {"no-scramble": NoFIM, "one_span": SingleSpanFIM, "multiple_span": MultipleSpanFIM}
    _POSIDS = {"none", "1d", "2d"}

    def __init__(self, 
                 msa_memmap_path=None,
                 msa_memmap_meta_path=None,
                 subset_path=None,
                 sample=False,
                 max_msa_len=-1,
                 reverse=False,
                 seed=42,
                 troubleshoot=False,
                 fim_strategy="no-scramble",
                 max_patches=5,
                 mask_fraction=0.2,
                 always_mask=False,
                 max_position_embeddings=2048,
                 max_seq_position_embeddings=512,
                 add_position_ids="1d", ):
        
        np.random.seed(seed)

        if msa_memmap_path:
            self.dataset = np.memmap(msa_memmap_path, dtype=np.int8, mode='r')
            self.dataset_meta = pd.read_csv(msa_memmap_meta_path)
            if subset_path:
                subset_ids = pd.read_csv(subset_path, header=None, names=['ID'])['ID'].tolist()
                self.dataset_meta = self.dataset_meta[self.dataset_meta['msa_id'].isin(subset_ids)]
        else:
            self.dataset = None

        self.sample = sample
        self.max_msa_len = max_msa_len
        self.reverse = reverse
        self.fim_strategy = fim_strategy
        if fim_strategy in ProteinMemmapDataset._FIM:
            self.fim = ProteinMemmapDataset._FIM[fim_strategy](max_patches=max_patches,
                                                             mask_fraction=mask_fraction,
                                                             always_mask=always_mask,
                                                             add_position_ids=add_position_ids != "none",
                                                             troubleshoot=troubleshoot)
        else:
            raise ValueError(f'Fill in the middle stragy "{fim_strategy}" not recognized.')
        
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_position_embeddings = max_seq_position_embeddings
        self.add_position_ids = add_position_ids

        self.troubleshoot = troubleshoot

    def __len__(self):
        # meta dataframe has one row for each MSA cluster
        return len(self.dataset_meta)

    def __getitem__(self, idx):
        # get all the sequences in the cluster
        sequences = self.get_sequences(idx)
        # get total number of sequences in the cluster and choose how many to sample
        orig_num_sequences = len(self.get_index_start_of_sequences(sequences))
        num_sequences = np.random.randint(1, orig_num_sequences + 1) if self.sample else orig_num_sequences
        # sample the sequences
        sequences, position_ids = self.sample_sequences(sequences, num_sequences)
        # with probability 0.5, reverse the sequences and move the last token to the front
        sequences, position_ids = self.reverse_sequences(sequences, position_ids) if (
                self.reverse and np.random.rand() > 0.5) else sequences, position_ids
        # limit the length of the MSA
        sequences = sequences[:self.max_msa_len] if self.max_msa_len > 0 else sequences
        if self.add_position_ids != "none":
            position_ids = position_ids[:self.max_msa_len] if self.max_msa_len > 0 else position_ids
        # convert to tensor
        sequences = torch.asarray(sequences, dtype=torch.int64)
        position_ids = torch.asarray(position_ids, dtype=torch.int64).clamp(0,
                                                                            self.max_position_embeddings - 1) if self.add_position_ids!="none" else None

        if self.troubleshoot:
            print(
                f"Cluster {idx} has {orig_num_sequences} sequences, of which {num_sequences} sampled now. Total MSA length: {len(sequences)}")
        if self.add_position_ids == "1d":
            return dict(input_ids=sequences, position_ids=position_ids, labels=sequences)
        if self.add_position_ids == "2d":
            seq_position_ids = (sequences == AA_TO_ID["<cls>"]).int().cumsum(-1).clamp(0,
                                                                                       self.max_seq_position_embeddings - 1).contiguous()
            return dict(input_ids=sequences, position_ids=position_ids, seq_position_ids=seq_position_ids,
                        labels=sequences)
        return dict(input_ids=sequences, labels=sequences)
    
    def get_msa_id(self, idx):
        """Get the MSA ID in the cluster with index `idx`."""
        cluster_meta = self.dataset_meta.iloc[idx]
        return cluster_meta.msa_id
    
    def get_idx_from_msa_id(self, msa_id):
        """Get `idx` with the MSA ID"""
        return self.dataset_meta[self.dataset_meta.msa_id == msa_id].index[0]

    def get_sequences(self, idx):
        """Get the sequences in the cluster with index `idx`."""
        cluster_meta = self.dataset_meta.iloc[idx]
        sequences = self.dataset[cluster_meta.Start : cluster_meta.End]
        return sequences

    def get_index_start_of_sequences(self, sequences):
        """Get the positions of the start of each sequence in the cluster."""
        return np.where(sequences == 0)[0]

    def reverse_sequences(self, sequence, position_ids=None):
        """Reverse the sequences and move the last token to the front."""
        sequence = sequence[::-1]
        if position_ids is not None:
            position_ids = position_ids[::-1]
        return np.concatenate([sequence[-1:], sequence[:-1]]), np.concatenate(
            [position_ids[-1:], position_ids[:-1]]) if position_ids is not None else None

    def sample_sequences(self, sequences, num_sequences, shuffle=True):
        """Sample `num_sequences` from the sequences in the cluster."""
        L = len(sequences)
        # get the indexes of the start of each sequence
        inds = self.get_index_start_of_sequences(sequences)
        # check that there are sequences in the cluster and that there are enough of them
        assert len(inds) > 0, "No sequences found in cluster."
        assert len(inds) >= num_sequences, "Not enough sequences in cluster."
        # sample n_sequences randomly from the sequences
        if shuffle:
            which_seqs = np.random.choice(np.arange(len(inds)), num_sequences, replace=False)
        else:
            which_seqs = np.arange(len(inds))[-num_sequences:]
        # get the tuples of start and end indexes of the sequences
        tuples = [(inds[i], inds[i + 1]) if i < len(inds) - 1 else (inds[i], L) for i in which_seqs]
        if self.troubleshoot:
            print(f"Sampled sequences: {tuples}")
        # concatenate the sequences
        sequences, position_ids = self.fim.apply(sequences, tuples)
        return sequences, position_ids



def make_dataloader(dataset):
    """Basic function to make a dataloader.
    """
    dataloader = DataLoader(dataset)
    return dataloader


class ProteinDataCollator(object):
    """
    Collate examples into a batch, and pad batch to a specified maximum sequence length,
    or to the longest sequence in the batch if max_sequence_length is None.
    """
    def __init__(self, max_sequence_length: Optional[int] = None):
        """
        Initialize the collator with an optional max_sequence_length.
        
        Args:
            max_sequence_length (Optional[int]): The maximum sequence length to pad/truncate to.
                                                 If None, pad to the longest sequence in the batch.
        """
        self.max_sequence_length = max_sequence_length

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "input_ids"))
        
        longest_seq = max(len(seq) for seq in input_ids)
        if self.max_sequence_length is None:
            max_len = longest_seq
        else:
            max_len = self.max_sequence_length

        input_ids = self.pad_sequences(input_ids, max_len, padding_value=AA_TO_ID["<pad>"])
        
        labels = self.pad_sequences(labels, longest_seq, padding_value=AA_TO_ID["<pad>"])
        labels = self.pad_sequences(labels, max_len, padding_value=-100)

        return_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(AA_TO_ID["<pad>"])
        )

        if "position_ids" in instances[0]:

            position_ids = [instance["position_ids"] for instance in instances]
            position_ids = self.pad_sequences(position_ids, max_len, padding_value=0)
            return_dict["position_ids"] = position_ids
            
            if "seq_position_ids" in instances[0]:
                seq_position_ids = [instance["seq_position_ids"] for instance in instances]
                seq_position_ids = self.pad_sequences(seq_position_ids, max_len, padding_value=0)
                return_dict["seq_position_ids"] = seq_position_ids

        return return_dict
        
    def pad_sequences(self, seqs, max_length, padding_value):
        # truncate long sequences (redundant, already done in __getitem__, maybe safe to remove)
        seqs = [seq[:max_length] for seq in seqs]

        # pad to same length
        seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=padding_value)

        # pad to max length
        padding = max_length - seqs.size(1)
        seqs = torch.nn.functional.pad(seqs, (0, padding), value=padding_value)

        return seqs