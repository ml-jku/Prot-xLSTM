import csv
import os

import numpy as np
from tqdm import tqdm

from protxlstm.utils import load_sequences_from_msa_file, tokenizer

def process_msa(msa_item):
    msa_name, msa_path = msa_item
    # Load an a3m file with all the context sequences
    msa = load_sequences_from_msa_file(msa_path)
    # Tokenize the sequences and concatenate them into a single array
    tokens = tokenizer(msa, concatenate=True)
    tokens = tokens.numpy()[0]
    return msa_name, tokens

def main(data_dir, output_dir):
    msa_paths = {k: os.path.join(data_dir, k, 'a3m/uniclust30.a3m') for k in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, k))}
    msa_items = list(msa_paths.items())

    dataset_dictionary = {}
    total_length = 0

    # First pass: calculate total length of all concatenated arrays
    for item in tqdm(msa_items):
        try:
            k, v = process_msa(item)
            dataset_dictionary[k] = v
            total_length += len(v)
        except:
            print(f"Error processing {item}")

    # Initialize the memmap array with the calculated total length
    memmap_path = os.path.join(output_dir, 'open_protein_set_memmap.dat')
    concatenated_array = np.memmap(memmap_path, dtype='int8', mode='w+', shape=(total_length,))

    with open(f'{output_dir}/open_protein_set_memmap_indices.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        csvwriter.writerow(['msa_id', 'Start', 'End'])
        
        start_index = 0
        for key, array in dataset_dictionary.items():
            end_index = start_index + len(array) - 1
            concatenated_array[start_index:end_index + 1] = array  # Write to memmap
            csvwriter.writerow([key, start_index, end_index])
            start_index = end_index + 1

    # Ensure the data is written to disk
    concatenated_array.flush()


if __name__ == "__main__":
    data_dir = 'data/a3m_files'
    output_dir = 'data/'
    main(data_dir, output_dir)



