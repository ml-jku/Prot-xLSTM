import torch
from tqdm import tqdm
import pickle
import os
import argparse
import json

from protxlstm.dataloaders import ProteinMemmapDataset
from protxlstm.generation import generate_sequence
from protxlstm.utils import (
    AA_TO_ID,
    load_model,
) 
from protxlstm.models.xlstm import xLSTMLMHeadModel
from protxlstm.models.mamba import MambaLMHeadModelwithPosids


def sample_sequences(dataset,
                     model,
                     family_idx,
                     params,
                     n_samples_per_family,
                     max_length=1000,
                     chunk_chunk_size=2**15,
                     save_path=None,
                     device="cuda:0"):
    """
    Function to sample sequences from the model. Given a dataset, a list of families (their indexes in the dataset)
    and a set of generating parameters, it generates `n_samples_per_family` sequences for each family and each parameter set.
    The function returns a dictionary with the following structure:
    gen_seqs = {family_idx: {parameters: {sequence: perplexity}}}
    The parameters are in a list of tuples with the following structure:    
    parameters_list = [(nr_seqs_ctx, temperature, top_k, top_p)]
    """        
    gen_seqs = {}
    gen_seqs[family_idx] = {}
    gen_seqs[family_idx][params] = {}
    print(f"Sampling sequences for family {family_idx} and parameters {params}.")

    n_seqs_ctx , temperature, top_k, top_p = params
    for _ in tqdm(range(n_samples_per_family)):
        # Sample the dataset to get the input
        data = dataset[family_idx]
        tokens = data["input_ids"][None,:].to(device)
        pos_ids = data["position_ids"][None,:].to(device)

        start_seqs = torch.argwhere(tokens[0]==0)[:,0].cpu().numpy()

        n_seqs_ctx = len(start_seqs) if len(start_seqs) < n_seqs_ctx else n_seqs_ctx
        L = start_seqs[n_seqs_ctx]+1
        context_tokens = tokens[:,:L]
        context_pos_ids = pos_ids[:,:L]
        is_fim={}

        # Generate the new sequence               
        output = generate_sequence(model,
                                context_tokens,
                                position_ids=context_pos_ids,
                                is_fim=is_fim,
                                max_length=(L+max_length),
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                return_dict_in_generate=True,
                                output_scores=True,
                                eos_token_id=torch.tensor([AA_TO_ID["<cls>"]]).to(device),
                                chunk_chunk_size=chunk_chunk_size,
                                device=device)
        
        # Get the perplexity of the generated sequence
        output_seq = output["generated"] 
        loss = torch.nn.functional.cross_entropy(torch.from_numpy(output["scores"]).permute(0, 2, 1),
                                                torch.from_numpy(output["generated_tokens"][0][None,:]))
        
        # save only sequences with length < max_length
        if len(output_seq[0]) < max_length:

            gen_seqs[family_idx][params][output_seq[0]] = {"perplexity": torch.exp(loss).item()}

    if save_path is not None:
        if not os.path.exists("evaluation/generation/generated_sequences"):
            os.mkdir("evaluation/generation/generated_sequences")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(f'{save_path}/{family_idx}_{params}_{n_samples_per_family}', "wb") as f:
            pickle.dump(gen_seqs, f)
        print(f"Sequences saved for family {family_idx} and parameters {params}")

    return gen_seqs

def generate_sequences(model_name,
                    checkpoint,
                    family_idxs=[],
                    parameters_list=[],
                    n_samples_per_family = 100,
                    chunk_size=1024,
                    chunk_chunk_size=2**15,
                    data_dir="data/",
                    device="cuda:0"
                    ):
    
    # Load the test dataset
    fim_strategy = "multiple_span"
    mask_fraction = 0.2

    dataset = ProteinMemmapDataset(
            msa_memmap_path=f"{data_dir}encoded_uniclust30f_int8_v2.dat",
            msa_memmap_meta_path=f"{data_dir}encoded_uniclust30f_int8_v2_memmap_indices.csv",
            subset_path=f"{data_dir}cluster_testing_set.txt",
            sample=False,
            max_msa_len=-1,
            reverse=False,
            seed=0,
            troubleshoot=False,
            fim_strategy=fim_strategy,
            always_mask=False,
            max_position_embeddings=2048,
            max_seq_position_embeddings=512,
            add_position_ids="1d",
            mask_fraction=mask_fraction
        )

    if model_name == "xlstm":
        model_class = xLSTMLMHeadModel
    elif model_name == "mamba":
        model_class = MambaLMHeadModelwithPosids

    save_path = f"evaluation/generation/generated_sequences/{checkpoint.split('/')[-1]}"

    if model_name == "xlstm":
        config_update_kwargs = {
                "mlstm_backend": "chunkwise_variable",
                "mlstm_chunksize": chunk_size,
                "mlstm_return_last_state": True
            }
    else:
        config_update_kwargs = {}

  
    #load the model
    model = load_model(checkpoint,
                    model_class=model_class,
                    device=device,
                    dtype=torch.bfloat16,
                    **config_update_kwargs,
                    )
    model = model.eval()
    print("Model loaded.")

    for family_idx in family_idxs:
        for params in parameters_list:
            params = tuple(params)
            if not os.path.exists(f'{save_path}/{family_idx}_{params}_{n_samples_per_family}'):
                gen_seqs = sample_sequences(
                        dataset=dataset,
                        model=model,
                        family_idx=family_idx,
                        params=params,
                        n_samples_per_family=n_samples_per_family,
                        chunk_chunk_size=chunk_chunk_size,
                        save_path=save_path,
                        device=device)
                
                print(f"Sampled {len(gen_seqs[family_idx][params])} valid sequences.")
            else:
                print(f"Sequences for family {family_idx} and parameters {params} already exist.")
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate sequences."
    )
    parser.add_argument("--model_name", type=str, help="Either 'xlstm' or 'mamba'.")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint.")
    parser.add_argument("--family_idxs", type=str, help="List of family indices.")
    parser.add_argument("--parameters_list", type=str, help="List of sampling parameters.")
    parser.add_argument("--n_samples_per_family", type=int, default=100, help="Number of sequences to sample per family and parameter set.")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for xLSTM context encoding.")
    parser.add_argument("--chunk_chunk_size", type=int, default=2*15, help="Length of context sequence part processed at once.")
    parser.add_argument("--data_dir", type=str, default="data/", help="Path to dataset.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device.")

    args = parser.parse_args()

    family_idxs = json.loads(args.family_idxs)
    parameters_list = json.loads(args.parameters_list)

    # Run sequence generation
    generate_sequences(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        family_idxs=family_idxs,
        parameters_list=parameters_list,
        n_samples_per_family=args.n_samples_per_family,
        chunk_size=args.chunk_size,
        chunk_chunk_size=args.chunk_chunk_size,
        data_dir=args.data_dir,
        device=args.device,
        )