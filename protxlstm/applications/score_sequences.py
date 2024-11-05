import argparse
import os
import pickle

from generation_utils.create_sequence_df import create_sequence_df
from generation_utils.score_hamming import score_hamming
from generation_utils.score_hmmer import score_hmmer
from generation_utils.score_structure import score_structure


def score_sequences(model_name,
                    family_idx,
                    num_sequences = 100,
                    data_dir = "data/"):
    
    if os.path.isfile(f"evaluation/generation/evaluations/{model_name}/sequence_df_{family_idx}"):
        with open(f"evaluation/generation/evaluations/{model_name}/sequence_df_{family_idx}", "rb") as f:
            sequence_df = pickle.load(f)
    else:
        sequence_df = create_sequence_df(model_name, family_idx, data_dir = data_dir, num_sequences = num_sequences)
        if not os.path.exists("evaluation/generation/evaluations/"):
            os.mkdir("evaluation/generation/evaluations/")
        if not os.path.exists(f"evaluation/generation/evaluations/{model_name}/"):
            os.mkdir(f"evaluation/generation/evaluations/{model_name}/")
        with open(f"evaluation/generation/evaluations/{model_name}/sequence_df_{family_idx}", "wb") as f:
            pickle.dump(sequence_df, f)

    if not "min_hamming" in sequence_df.columns:
        sequence_df = score_hamming(sequence_df, family_idx, data_dir)
        with open(f"evaluation/generation/evaluations/{model_name}/sequence_df_{family_idx}", "wb") as f:
            pickle.dump(sequence_df, f)

    if not "score_gen" in sequence_df.columns:
        sequence_df = score_hmmer(sequence_df, family_idx, data_dir)
        with open(f"evaluation/generation/evaluations/{model_name}/sequence_df_{family_idx}", "wb") as f:
            pickle.dump(sequence_df, f)

    if not "ptm" in sequence_df.columns:
        sequence_df = score_structure(sequence_df, family_idx)
        with open(f"evaluation/generation/evaluations/{model_name}/sequence_df_{family_idx}", "wb") as f:
            pickle.dump(sequence_df, f)

    return sequence_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate sequences."
    )
    parser.add_argument("--model_name", type=str, help="Either 'xlstm' or 'mamba'.")
    parser.add_argument("--family_idx", type=int, help="Family index.")
    parser.add_argument("--num_sequences", type=int, default=100, help="Number of sequences.")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Path to dataset.")

    args = parser.parse_args()

    sequence_df = score_sequences(args.model_name, args.family_idx, args.num_sequences, args.data_dir)
