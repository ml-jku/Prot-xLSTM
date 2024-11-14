import numpy as np
import pickle
import pandas as pd

from protxlstm.dataloaders import ProteinMemmapDataset
from protxlstm.utils import decode_sequence, reorder_masked_sequence


def create_sequence_df(model_name, family_idx, parameters_list=None, num_sequences = 100, data_dir="./data/"):

    #load dataset
    dataset = ProteinMemmapDataset(
            msa_memmap_path=f"{data_dir}open_protein_set_memmap.dat",
            msa_memmap_meta_path=f"{data_dir}open_protein_set_memmap_indices.csv",
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
    
    family_id = list(dataset.dataset_meta["msa_id"])[family_idx]

    if model_name == "natural":

        data = dataset[family_idx]
        sequence_df = pd.DataFrame(columns=["family", "family_id", "sequence", "sequence_length"])
        tokens = data["input_ids"][None,:]
        all_context = decode_sequence(tokens[0].cpu().numpy())
        list_sequences_msa = [reorder_masked_sequence(elem+"<cls>") for elem in all_context.split("<cls>")[1:-1]]

        rd_idxs = np.random.choice(len(list_sequences_msa), num_sequences, replace=False)
        natural_sequences = [seq for i, seq in enumerate(list_sequences_msa) if i in rd_idxs]

        df_dict = {"family": [family_idx]*len(natural_sequences), 
                    "family_id": [family_id]*len(natural_sequences),
                    "sequence": natural_sequences,
                    "sequence_length": [len(seq) for seq in natural_sequences]}

        sequence_df = pd.concat([sequence_df, pd.DataFrame(df_dict)], ignore_index = True)

    else:
        
        sequence_df = pd.DataFrame(columns=["family", "family_id",  "n_seqs_ctx", "temperature", "top_k", "top_p", "original_sequence", "sequence", "sequence_length", "perplexity"])
        
        if parameters_list is None:
            parameters_list = [(10,1.,10,1.), (10,1.,15,1.), (10,1.,10,0.95), (10,0.9,10,0.95), (10,0.8,10,0.9),
                    (100,1.,10,1.), (100,1.,15,1.), (100,1.,10,0.95), (100,0.9,10,0.95), (100,0.8,10,0.9),
                    (500,1.,10,1.), (500,1.,15,1.), (500,1.,10,0.95), (500,0.9,10,0.95), (500,0.8,10,0.9),
                    (1000,1.,10,1.), (1000,1.,15,1.), (1000,1.,10,0.95), (1000,0.9,10,0.95), (1000,0.8,10,0.9),
                    (-1,1.,10,1.), (-1,1.,15,1.), (-1,1.,10,0.95), (-1,0.9,10,0.95), (-1,0.8,10,0.9)]
        
        for param in parameters_list:
            n_seqs_ctx, temperature, top_k, top_p = param

            with open(f"evaluation/generation/generated_sequences/{model_name}/{family_idx}_{param}_{num_sequences}", "rb") as f:
                gen_seqs = pickle.load(f)

            original_sequences =  list(gen_seqs[family_idx][param].keys())
            reordered_sequences = [reorder_masked_sequence(seq) for seq in original_sequences]
            perplexities = [gen_seqs[family_idx][param][seq]["perplexity"] for seq in original_sequences]
            df_dict = {"family": [family_idx]*len(original_sequences), 
                        "family_id": [family_id]*len(original_sequences),
                        "n_seqs_ctx": [n_seqs_ctx]*len(original_sequences),
                        "temperature": [temperature]*len(original_sequences),
                        "top_k": [top_k]*len(original_sequences),
                        "top_p": [top_p]*len(original_sequences),
                        "original_sequence": original_sequences,
                        "sequence": reordered_sequences,
                        "sequence_length": [len(seq) for seq in reordered_sequences],
                        "perplexity": perplexities
                        }

            sequence_df = pd.concat([sequence_df, pd.DataFrame(df_dict)], ignore_index = True)

    return sequence_df

