import argparse
import os
import re

import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from protxlstm.applications.fitness_prediction import single_mutation_landscape_mamba, single_mutation_landscape_xlstm, single_mutation_landscape_retrieval
from protxlstm.applications.msa_sampler import sample_msa
from protxlstm.models.mamba import MambaLMHeadModelwithPosids
from protxlstm.models.xlstm import xLSTMLMHeadModel
from protxlstm.utils import AA_TO_ID, load_sequences_from_msa_file, load_model

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--name", type=str, default="protxlstm_102M_60B",
        help="Model name."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/protxlstm_102M_60B",
        help="Path to model checkpoint folder."
    )
    parser.add_argument(
        "--base_folder", type=str, default="data/proteingym/",
        help="Path to base directory for `MS_reference_file_path`, `DMS_data_folder`, `output_scores_folder`, `MSA_folder` and `retrieval_MSA_folder`."
    )
    parser.add_argument(
        "--DMS_reference_file_path", type=str, default="substitutions/DMS_substitutions.csv",
        help="Path to DMS reference file."
    )
    parser.add_argument(
        "--DMS_data_folder", type=str, default="substitutions/DMS_ProteinGym_substitutions/",
        help="Path to DMS substitutions folder."
    )
    parser.add_argument(
        "--DMS_index", type=int, default=1,
        help="Index of DMS assay.")

    parser.add_argument(
        "--output_scores_folder", type=str, default="zero_shot_substitutions_scores",
        help="Path to output folder."
    )
    parser.add_argument(
        "--MSA_folder", type=str, default="msa_files/colabfold/",
        help="Path to MSA folder used to sample context sequences."
    )
    parser.add_argument(
        "--context_length", type=int, default=2**17,
        help="Maximum number of context tokens."
    )
    parser.add_argument(
        "--max_context_sequences", type=int, default=200,
        help='Maximum number of context sequences with the limit of `context_length`. If `0` all sequences in MSA are used.'
    )
    parser.add_argument(
        "--n_replicates", type=int, default=3,
        help='Number replicates. For each replicate, a different random sampling of the context is performed.'
    )
    parser.add_argument(
        "--retrieval_alpha", type=float, default=0,
        help="If retrieval_alpha > 0, ensemble Prot-xLSTM predictions with independent-site model predictions."
    )
    parser.add_argument(
        "--retrieval_MSA_folder", type=str, default="data/proteingym/msa_files/proteingym/",
        help = 'Path to MSA folder used for independent-site model predictions.'
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite the output file."
    )

    args = parser.parse_args()
    args.DMS_reference_file_path = os.path.join(args.base_folder, args.DMS_reference_file_path)
    args.DMS_data_folder = os.path.join(args.base_folder, args.DMS_data_folder)
    args.MSA_folder = os.path.join(args.base_folder, args.MSA_folder)
    args.output_scores_folder = os.path.join(args.base_folder, args.output_scores_folder)
    os.makedirs(args.output_scores_folder, exist_ok=True)

    if args.retrieval_alpha > 0:
        args.retrieval_MSA_folder = os.path.join(args.base_folder, args.MSA_folder)
        args.retrieval_output_scores_folder = os.path.join(args.base_folder, args.output_scores_folder + f"_r{args.retrieval_alpha:.2f}")
        os.makedirs(args.retrieval_output_scores_folder, exist_ok=True)

    return args

def load_model_(args):
    # Load model
    if 'mamba' in args.name.lower():
        model = load_model(
            args.checkpoint,
            model_class=MambaLMHeadModelwithPosids,
            device="cuda",
            dtype=torch.float32,
            checkpoint_mixer=True,
        ).eval()
    elif 'xlstm' in args.name.lower():
        config_update_kwargs = {
                "mlstm_backend": "chunkwise_variable",
                "mlstm_chunksize": 1024,
                "mlstm_return_last_state": True
            }
        model = load_model(
            args.checkpoint,
            model_class=xLSTMLMHeadModel,
            device="cuda",
            dtype=torch.float32,
            **config_update_kwargs,
        ).eval()
    return model

def load_msa(msa_folder, df):
    """Load MSAs"""
    if 'colab' in msa_folder.lower():
        # MSA from colabfold
        msa_filepath = os.path.join(msa_folder, df["UniProt_ID"] + ".a3m")
        msa_sequences = load_sequences_from_msa_file(msa_filepath)
        msa_weights_path = os.path.join(msa_folder, "weights", df["UniProt_ID"] + ".npy")
    elif 'proteingym' in msa_folder.lower():
        # MSA from proteingym
        msa_filepath = os.path.join(msa_folder, df["MSA_filename"])
        msa_sequences = load_sequences_from_msa_file(msa_filepath)
        msa_weights_path = os.path.join(msa_folder, "weights", df["MSA_weights_file"])
    
    # Crop MSA to section speficied by proteingym
    return [msa.upper()[df["MSA_start"] - 1 : df["MSA_end"]] for msa in msa_sequences], msa_weights_path

def extra_non_retrieval_path(original_path):
    # Get the folder part of the path
    folder, file_name = os.path.split(original_path)
    # Use a regular expression to remove the _r<some_float> from the folder name
    new_folder = re.sub(r'_r[\d.]+', '', folder)
    # Join the new folder name with the file name to get the desired path
    return os.path.join(new_folder, file_name)    

if __name__ == "__main__":

    args = parse_args()

    # Get DMS id
    ref_series = pd.read_csv(args.DMS_reference_file_path).iloc[args.DMS_index]
    dms_id = ref_series["DMS_id"]
    variants_filename = ref_series["DMS_filename"]

    # If output file exist and now overwrite, exit
    output_folder = args.output_scores_folder if args.retrieval_alpha == 0 else args.retrieval_output_scores_folder
    if not args.overwrite and os.path.exists(os.path.join(output_folder, variants_filename)):
        exit()

    # Get WT
    print('Processing', dms_id)
    uniprot_id = ref_series["UniProt_ID"]
    msa_start = int(ref_series["MSA_start"])
    msa_end = int(ref_series["MSA_end"])
    wt_sequence = ref_series["target_seq"][msa_start - 1 : msa_end]
    print('Sequence length', len(wt_sequence))

    # Get variations
    variants_df = pd.read_csv(os.path.join(args.DMS_data_folder, variants_filename))
    mutations = variants_df["mutant"].values
    mutations_set = [(mut, int(mut[1:-1]) - msa_start, AA_TO_ID[mut[-1]]) for mutation in variants_df["mutant"].values for mut in mutation.split(':')]
    single_mutations = pd.DataFrame(mutations_set, columns=['mutation', 'position', 'mutation_idx']).drop_duplicates()
    n_mutated_positions = single_mutations.position.nunique()
    print(f"Number of mutations | single mutations | mutation positions: {len(mutations)} | {len(single_mutations)} | {n_mutated_positions}")

    non_retrieval_score_file = os.path.join(args.output_scores_folder, variants_filename)
    if args.overwrite or not os.path.exists(non_retrieval_score_file):

        # Load model
        model = load_model_(args)
        
        # Load MSA
        msa_sequences, msa_weights_path = load_msa(args.MSA_folder, ref_series)
    
        for i in tqdm(range(args.n_replicates), desc='Replicates'):

            # sample context sequences
            context_sequences = sample_msa(msa_sequences, msa_weights_path, context_length=args.context_length, max_context_sequences=args.max_context_sequences, seed=i) + [wt_sequence]
            n_context_sequences = len(context_sequences)
            n_context_tokens = len(context_sequences)*len(msa_sequences[0])
            print(f"Number of context sequences: {n_context_sequences} ({n_context_tokens:,} tokens)")

            # single mutation landscape
            if 'mamba' in args.name.lower():
                single_mutations, all_logits = single_mutation_landscape_mamba(model, single_mutations, context_sequences)
            else:
                single_mutations, all_logits = single_mutation_landscape_xlstm(model, single_mutations, context_sequences)
            
            if args.n_replicates > 1:
                single_mutations.rename(columns={'effect': f'effect_{i}'}, inplace=True)

            torch.cuda.empty_cache()

        # Average effect over replicas
        if args.n_replicates > 1:
            effect_columns = [col for col in single_mutations.columns if col.startswith('effect_')]
            single_mutations['effect'] = single_mutations[effect_columns].mean(axis=1)
            single_mutations.drop(columns=effect_columns, inplace=True)

        # Get predicted score per mutated sequence (for multiple mutation sum of single mutations)
        variants_df['predicted_score'] = variants_df['mutant'].apply(
            lambda x : single_mutations.loc[single_mutations.mutation.isin(x.split(':')), 'effect'].sum())
    
        # Compute Spearman and save predictions
        rho = spearmanr(variants_df['DMS_score'], variants_df['predicted_score'])[0]
        print(f'{dms_id} Spearman: {rho:.3f}')     
        variants_df.to_csv(non_retrieval_score_file, index=False)

    if args.retrieval_alpha > 0:
        # Load non-retrieval scores
        variants_df = pd.read_csv(non_retrieval_score_file)

        # Load MSAs
        msa_sequences, msa_weights_path = load_msa(args.retrieval_MSA_folder, ref_series)

        # Independant-site model scores
        single_mutations = single_mutation_landscape_retrieval(single_mutations, msa_sequences, msa_weights_path)        
        
        variants_df['retrieval_score'] = variants_df['mutant'].apply(
            lambda x : single_mutations.loc[single_mutations.mutation.isin(x.split(':')), 'retrieval_effect'].sum())
        variants_df['predicted_score'] = (
            (1 - args.retrieval_alpha) * variants_df['predicted_score'].astype('float') +
            args.retrieval_alpha * variants_df['retrieval_score'].astype('float')
        )                       
        variants_df.to_csv(os.path.join(args.retrieval_output_scores_folder, variants_filename), index=False)
        rho = spearmanr(variants_df['DMS_score'], variants_df['predicted_score'])[0]
        print(f'{dms_id} Spearman (w/ retrieval - alpha {args.retrieval_alpha}): {rho:.3f}')