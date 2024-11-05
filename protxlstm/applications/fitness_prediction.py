import os
import numpy as np
import pandas as pd

import torch
from tqdm.auto import tqdm

from protxlstm.applications.msa_sampler import MSASampler
from protxlstm.generation import generate_sequence
from protxlstm.utils import AA_TO_ID, tokenizer, ID_TO_AA


def precompute_context_state(model, sequences, chunk_chunk_size=2**15):
    """
    Precompute the output states for a fixed context that remains the same across generations.
    Returns the hidden states to continue generation later.
    """
    device = next(model.parameters()).device

    input_ids, pos_ids = prepare_context(sequences)
    state = None
    
    for chunk in range(input_ids.shape[1]//chunk_chunk_size+1):

        start_idx = chunk*chunk_chunk_size
        end_idx = min((chunk+1)*chunk_chunk_size, input_ids.shape[1])

        if start_idx == end_idx:
            pass
        
        else:
            input_ids_chunk = input_ids[:, start_idx:end_idx].to(device)
            pos_ids_chunk = pos_ids[:, start_idx:end_idx].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids_chunk,
                                position_ids=pos_ids_chunk,
                                state=state,
                                output_hidden_states=True,
                                return_dict=True)
                state = outputs.state

    # Return the hidden states for reuse
    return state

def prepare_context(sequences):
    tokenized_sequences = tokenizer(sequences, concatenate=False)
    pos_ids = torch.cat([torch.arange(0, len(seq), dtype=torch.int64) for seq in tokenized_sequences], 0)[None, :]
    input_ids = torch.cat(tokenized_sequences, 0)[None, :].to(torch.int64)
    return input_ids, pos_ids

def prepare_single_mutation_target(target, mut_pos):

    pos_ids = torch.arange(target.shape[1], dtype=torch.int64)[None,:] # default position ids
    t = torch.ones((target.shape[0], 1), dtype=torch.int64)
    new_target = torch.cat([
        target[:,:mut_pos], # WT sequence until mutated position
        AA_TO_ID["<mask-1>"] * t, # Mask token at the muated position
        target[:,mut_pos+1:],  # WT sequence after mutated position
        AA_TO_ID["<eos>"] * t, # End of sequence token
        AA_TO_ID["<mask-1>"] * t, # Mask token
        ], dim=1)
    new_pos_ids = torch.cat([
        pos_ids,
        0 * t, # end of sequence
        mut_pos * t, # mutation position
        ], dim=1)

    is_fim_dict = { AA_TO_ID["<mask-1>"] : pos_ids[:,mut_pos].squeeze().item()}

    return new_target, new_pos_ids, is_fim_dict

def single_mutation_landscape_xlstm(model, single_mutations, context_sequences, chunk_chunk_size=2**15):

    device = next(model.parameters()).device

    # Tokenize WT target sequence    
    wt_tokens = tokenizer([context_sequences[-1]], concatenate=True)
    
    # Precompute hidden state of context
    context_state = precompute_context_state(model, context_sequences, chunk_chunk_size=chunk_chunk_size)

    mutation_positions = sorted(single_mutations.position.unique())    
    all_logits = np.zeros((len(mutation_positions), 20))

    # Iterate over all mutated positions
    for i, pos in tqdm(enumerate(mutation_positions), total=len(mutation_positions), desc="Generating mutational landscape"): # This loop can be parallelized
        
        # Prepare target
        wt_aa_id = wt_tokens[0, pos+1].int().item() # wild type AA index
        target_tokens, target_pos_ids, _ = prepare_single_mutation_target(wt_tokens, pos+1)

        with torch.no_grad():
            outputs = model(input_ids=target_tokens.to(device),
                            position_ids=target_pos_ids.to(device),
                            state=context_state,
            )

        # Extact logits and compute mutational effect
        logits = outputs.logits.clone().detach()  # Raw logits
        logits_mut = logits[0, -1, 4:24].log_softmax(-1)  # Log-softmax for mutation prediction: (4-24) correspond to natural NNs
        mut_effects = logits_mut - logits_mut[wt_aa_id - 4]  # Subtract log probability of ground truth
        all_logits[i,:] = logits_mut.cpu()
        single_mutations.loc[single_mutations.position == pos, 'effect'] = single_mutations.loc[single_mutations.position == pos, 'mutation_idx'].apply(lambda x : mut_effects[x-4].item())
    
    return single_mutations, all_logits

def single_mutation_landscape_mamba(model, single_mutations, context_sequences):

    # Prepare context sequences
    context_tokens, context_pos_ids = prepare_context(context_sequences)

    # Tokenize WT target sequence
    wt_tokens = tokenizer([context_sequences[-1]], concatenate=True)

    mutation_positions = sorted(single_mutations.position.unique())    
    all_logits = np.zeros((len(mutation_positions), 20))
    
    # Iterate over all mutated positions
    for i, pos in tqdm(enumerate(mutation_positions), total=len(mutation_positions), desc="Generating mutational landscape"): # This loop can be parallelized
        
        # Prepare target
        wt_aa_id = wt_tokens[0, pos+1].int().item() # wild type AA index
        target_tokens, target_pos_ids, is_fim_dict = prepare_single_mutation_target(wt_tokens, pos+1)
       
        # Merge context and target
        device = next(model.parameters()).device
        context_tokens = torch.cat([context_tokens, target_tokens], dim=1).to(device)
        context_pos_ids = torch.cat([context_pos_ids, target_pos_ids], dim=1).to(device)

        # Generate fim-token prediction
        output = generate_sequence(
                model,
                context_tokens,
                position_ids=context_pos_ids,
                is_fim=is_fim_dict,
                max_length=1,
                temperature=1.0,
                top_k=0,
                top_p=0.0,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=AA_TO_ID["<cls>"],
                device=device
            )
        
        # Extact logits and compute mutational effect
        logits = torch.tensor(output["scores"])  # Raw logits
        logits_mut = logits[0, 0, 4:24].log_softmax(-1)  # Log-softmax for mutation prediction: (4-24) correspond to natural NNs
        mut_effects = logits_mut - logits_mut[wt_aa_id - 4]  # Subtract log probability of ground truth
        all_logits[i,:] = logits_mut.cpu()

        single_mutations.loc[single_mutations.position == pos, 'effect'] = single_mutations.loc[single_mutations.position == pos, 'mutation_idx'].apply(lambda x : mut_effects[x-4].item())
    
    return single_mutations, all_logits

def single_mutation_landscape_retrieval(single_mutations, msa_sequences, msa_weights_path):

    # One-hot encode MSA sequences
    msa_tokens = np.array([[AA_TO_ID[aa.upper()] for aa in seq] for seq in msa_sequences])
    one_hot_tokens = np.zeros((len(msa_tokens), len(msa_tokens[0]), 40))
    one_hot_tokens[np.arange(len(msa_tokens))[:, None], np.arange(len(msa_tokens[0])), msa_tokens] = 1

    #Load/compute weights
    if os.path.exists(msa_weights_path):
        weights = np.load(msa_weights_path)
    else:
        sampler = MSASampler(0.98, 0.7)
        weights = sampler.get_weights(msa_tokens)[1]
        np.save(msa_weights_path, weights)
    assert one_hot_tokens.shape[0] == weights.shape[0]

    # Apply sequence weights, normalize amino acid probabilities per position, and convert to a PyTorch tensor.
    one_hot_tokens = one_hot_tokens * weights[:, None, None]
    one_hot_tokens = one_hot_tokens.sum(0)
    one_hot_tokens = one_hot_tokens[:, 4:24] + 1 / len(msa_sequences)
    one_hot_tokens_sum = one_hot_tokens.sum(-1)
    one_hot_tokens = one_hot_tokens / one_hot_tokens_sum[:, None]
    one_hot_tokens = torch.tensor(one_hot_tokens).float()    

    # Compute mutational effects
    wild_type = msa_tokens[0]
    logits = one_hot_tokens.log()
    logits = logits - logits[torch.arange(len(logits)), wild_type - 4][:, None]

    single_mutations['retrieval_effect'] = single_mutations.apply(
        lambda row: logits[row['position'], row['mutation_idx'] - 4].item(), axis=1)
    
    return single_mutations


def create_mutation_df(sequence, mutation_positions):
    """
    Generate a DataFrame containing all possible mutations at specified positions in a sequence.

    Args:
        sequence (str): The original sequence to mutate.
        mutation_positions (list of int): List of positions to mutate (1-based index).

    Returns:
        pd.DataFrame:
            - 'mutation': formatted mutation string (e.g., 'A10G' for Ala at position 10 to Gly).
            - 'position': 0-based position in the sequence.
            - 'mutation_idx': numeric index for the mutation.
    """
    
    AAs = {k: v for k, v in ID_TO_AA.items() if 4 <= k <= 23}
    mutation_data = []
    for position in mutation_positions:
        wt = sequence[position - 1] 
        for idx, aa in AAs.items():
            mutation = f"{wt}{position}{aa}"
            mutation_data.append({'mutation': mutation, 'position': position - 1, 'mutation_idx': idx})
    return pd.DataFrame(mutation_data)