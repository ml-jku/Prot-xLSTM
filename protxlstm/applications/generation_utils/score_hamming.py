import numpy as np
from tqdm import tqdm
import pandas as pd
from Bio import Align

from protxlstm.dataloaders import ProteinMemmapDataset
from protxlstm.utils import decode_sequence, reorder_masked_sequence


aligner = Align.PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 1
aligner.mismatch_score = -1
aligner.open_gap_score = -1
aligner.extend_gap_score = -1

def align_sequences(ref_seq, query_seq, print_alignments=False):
    def hamming_str(s1,s2):
        assert len(s1) == len(s2)
        return sum(np.array(list(s1)) != np.array(list(s2)))/len(s1)
    alignments = aligner.align(ref_seq, query_seq)
    if print_alignments:
        print("Score = %.1f:" % alignments[0].score)
        print(alignments[0])
    return hamming_str(alignments[0][0], alignments[0][1]), alignments[0][0], alignments[0][1]


def score_hamming(sequence_df, family_idx, data_dir = f"./data/"):

    assert len(set(list(sequence_df["family"]))) == 1 and sequence_df["family"].iloc[0] == family_idx

    #load dataset
    dataset = ProteinMemmapDataset(
            msa_memmap_path=f"{data_dir}/encoded_uniclust30f_int8_v2.dat",
            msa_memmap_meta_path=f"{data_dir}/encoded_uniclust30f_int8_v2_memmap_indices.csv",
            subset_path=f"{data_dir}/cluster_testing_set.txt",
            sample=False,
            max_msa_len=-1,
            reverse=False,
            seed=0,
            troubleshoot=False,
            fim_strategy="multiple_span",
            always_mask=False,
            max_position_embeddings=2048,
            max_seq_position_embeddings=512,
            add_position_ids="1d",
            mask_fraction=0.2,
            max_patches=5,
        )
    
    # Select a sample of the dataset to be the input
    data = dataset[family_idx]
    tokens = data["input_ids"][None,:]
    all_context = decode_sequence(tokens[0].cpu().numpy())
    list_sequences_msa = [reorder_masked_sequence(elem+"<cls>") for elem in all_context.split("<cls>")[1:-1]]

    # sequence_df["hamming"] = pd.Series(dtype=object)
    sequence_df["min_hamming"] = pd.Series()
    sequence_df["median_hamming"] = pd.Series()
    sequence_df["mean_hamming"] = pd.Series()
    sequence_df["std_hamming"] = pd.Series()

    for seq in tqdm(list(sequence_df["sequence"])):

        all_hamming = []
        for ctx_seq in list_sequences_msa:
            if ctx_seq == seq:
                continue
            else:
                hamming, _, _ = align_sequences(ctx_seq, seq , print_alignments=False)
                all_hamming.append(hamming)

        # sequence_df.loc[sequence_df["sequence"] == seq, "hamming"] = [all_hamming]
        sequence_df.loc[sequence_df["sequence"] == seq, "min_hamming"] = np.min(all_hamming)
        sequence_df.loc[sequence_df["sequence"] == seq, "median_hamming"] = np.median(all_hamming)
        sequence_df.loc[sequence_df["sequence"] == seq, "mean_hamming"] = np.mean(all_hamming)
        sequence_df.loc[sequence_df["sequence"] == seq, "std_hamming"] = np.std(all_hamming)

    return sequence_df
        