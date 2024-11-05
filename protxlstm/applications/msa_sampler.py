# Original code from ProtMamba under Apache License 2.0.
#
# Modifications made by Niklas Schmidinger, Lisa Schneckenreiter and Sohvi Luukkonen 
#   - Modify handling of weights in `MSASampler`

import math
import os
from typing import Optional, Callable

import numpy as np
import torch

from protxlstm.utils import AA_TO_ID


def compute_hamming_csim_torch(
        seqs: torch.Tensor,
        ungapped_msa: torch.Tensor,
        gap_token: int,
        gap_token_mask: int,
) -> torch.Tensor:
    return (seqs.unsqueeze(1) == ungapped_msa).sum(dim=2)

def _compute_homology_weights(
        ungapped_msa: np.ndarray,
        gap_token: int,
        gap_token_mask: int,
        theta: float,
        hamming_csim_func: Callable,
        max_memory: int = 20,
        can_use_torch: bool = True,
) -> np.ndarray:
    use_torch = can_use_torch and torch.cuda.is_available()
    if use_torch:
        hamming_csim_func = compute_hamming_csim_torch
    batch_size = math.floor(
        2
        * 1024
        * 1024
        * 1024
        / (ungapped_msa.shape[0] * ungapped_msa.shape[1])
        * max_memory
        / 40
    )

    batch_size = 1 if batch_size == 0 else batch_size

    neighbors = []
    if not use_torch:
        masked_ungapped_msa = ungapped_msa.copy()
    else:
        ungapped_msa = torch.from_numpy(ungapped_msa).byte().cuda()
        masked_ungapped_msa = ungapped_msa.clone()
    masked_ungapped_msa[masked_ungapped_msa == gap_token] = gap_token_mask
    for b_start in range(0, len(ungapped_msa), batch_size):
        b_end = b_start + batch_size
        seqs = ungapped_msa[b_start:b_end]

        sim = hamming_csim_func(
            seqs=seqs,
            ungapped_msa=masked_ungapped_msa,
            gap_token=gap_token,
            gap_token_mask=gap_token_mask,
        )
        if not use_torch:
            sim = sim / (seqs != gap_token).sum(axis=1, keepdims=True)
            d = 1 - sim
            d = d.clamp(0, 1)
            this_neighbors = (d <= theta).sum(axis=1)
        else:
            sim = sim / (seqs != gap_token).sum(dim=1, keepdim=True)
            d = 1 - sim
            # fillna
            d[torch.isnan(d)] = 0
            d = d.clamp(0, 1)
            this_neighbors = (d <= theta).sum(dim=1).cpu()
        neighbors.append(this_neighbors)
    return np.concatenate(neighbors)

def compute_homology_weights(
        ungapped_msa: np.ndarray,
        theta: float = 0.2,
        gap_token: int = AA_TO_ID["-"],
        gap_token_mask: int = 255,
        hamming_csim_func: Callable = compute_hamming_csim_torch,
) -> tuple[int, np.ndarray]:
    """
    Calculate the effective number of sequences and sampling probability for the NEIGHBORS and NEIGHBORS_NO_LIMIT sampling methods using numpy.

    Parameters:

        ungapped_msa (np.ndarray): The MSA (from .fa).
        theta (float, optional): A parameter used to determine the similarity between sequences. Default is 0.2.
        gap_token (int, optional): The token representing gaps in the (Uniprot21 encoded) MSA. Default is 20.
        gap_token_mask (int): token for masking gaps. should be a token not representing any other value.

    Returns:

        tuple[int, np.ndarray]: A tuple containing the effective number of sequences and the sampling probability for each sequence in the MSA.
    """
    neighbors = _compute_homology_weights(
        ungapped_msa=ungapped_msa,
        gap_token=gap_token,
        gap_token_mask=gap_token_mask,
        theta=theta,
        hamming_csim_func=hamming_csim_func,
    )
    n_eff = np.sum(1 / neighbors)

    p = 1 / neighbors
    p /= np.sum(p)
    return n_eff, p

class MSASampler:

    def __init__(self, max_similarity, max_dissimilarity, force_include_first=True):
        self.max_similarity = max_similarity
        self.max_dissimilarity = max_dissimilarity
        self.force_include_first = force_include_first
        self.theta = 0.2

    def _get_sim_filtered_idxs(self, msa: np.ndarray) -> np.ndarray:
        nonnormalized_sim = (msa == msa[[0]]).sum(axis=1)
        normfactor = msa.shape[1]
        norm_sim = nonnormalized_sim / normfactor

        assert (norm_sim.min() >= 0) and (norm_sim.max() <= 1)
        dsim = 1 - norm_sim

        max_sim_filter = norm_sim <= self.max_similarity
        max_dissim_filter = dsim <= self.max_dissimilarity
        return np.where(max_sim_filter & max_dissim_filter)[0]

    def get_weights(
            self, msa: np.ndarray,
    ) -> tuple[Optional[float], Optional[np.ndarray]]:
        return compute_homology_weights(
            ungapped_msa=msa,
            theta=self.theta,
            gap_token_mask=255,

        )

    def get_sample_idxs(
            self,
            msa: np.ndarray,
            size: int = 1,
            random = False,
            msa_weights_path = None,
            seed = 0,
    ) -> np.ndarray:
        
        np.random.seed(seed)
        
        if random:
            return np.random.choice(len(msa), replace=False, size=size) if len(msa) >= size else np.arange(len(msa))
        
        msa = np.array([[AA_TO_ID[aa] for aa in seq.upper()][:len(msa[0])] for seq in msa], dtype=np.uint8)

        if msa_weights_path and os.path.exists(msa_weights_path):
            weights = np.load(msa_weights_path)
        elif msa_weights_path:
            os.makedirs(os.path.dirname(msa_weights_path), exist_ok=True)
            _, weights = self.get_weights(
                msa=msa,
            )
            np.save(msa_weights_path, weights)
        else:
            _, weights = self.get_weights(
                msa=msa,
             )


        original_msa_sample_idxs = np.arange(len(msa))
        sample_idxs = self._get_sim_filtered_idxs(msa)
        original_msa_sample_idxs = original_msa_sample_idxs[sample_idxs]

        if self.force_include_first:
            original_msa_sample_idxs = np.concatenate(
                [[0], original_msa_sample_idxs[original_msa_sample_idxs != 0]]
            )
        return np.random.choice(len(msa), replace=False, size=size, p=weights / weights.sum()) if len(msa) >= size else original_msa_sample_idxs
    
def sample_msa(msa_sequences, msa_weights_path=None, context_length=200_000, max_context_sequences=200, seed=0, sort=True):
    """Sample MSA sequences for the context"""    
    n_sequences = min( context_length // len(msa_sequences[0]), len(msa_sequences) if max_context_sequences == 0 else max_context_sequences ) - 1 
    sampler = MSASampler(0.98, 0.7, force_include_first=False)
    sample_idx = sampler.get_sample_idxs(
        msa_sequences, size=n_sequences, msa_weights_path=msa_weights_path, seed=seed
        )
    
    # Sort sequences from least similar to most similar and add wild type target sequence
    if sort:
        context_sequences = [msa_sequences[i] for i in sample_idx][::-1]
    
    return context_sequences