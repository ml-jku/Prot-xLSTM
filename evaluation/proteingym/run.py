import os
import numpy as np
import pandas as pd
import subprocess
from tqdm.auto import tqdm
from scipy.stats import spearmanr

models = ['protxlstm_26M_30B', 'protmamba_28M_30B', 'protxlstm_102M_60B', 'protmamba_107M_195B']

nct = 200_000
ncs = 200
nr = 3
ra = 0

DMS_file = 'data/proteingym/substitutions/DMS_substitutions.csv'
df = pd.read_csv(DMS_file).head(10)
# df = df.sample(frac=1)

problematics = ['BRCA2_HUMAN_Erwood_2022_HEK293T'] # One target fails on Mamba models


for model in models:
    output_dir = f"results/{model}_nct{nct}_ncs{ncs}_{nr}"
    
    for seq in tqdm(df.DMS_id):
        if seq in problematics and 'mamba' in model:
            continue
        i = df[df.DMS_id == seq].index[0]
        
        if os.path.exists(f"{output_dir}/{seq}.csv"):
            continue
        
        cmd = f"python evaluation/proteingym/score.py --DMS_index {i} --name {model} \
            --context_length {nct} --max_context_sequences {ncs} \
            --n_replicates {nr} --output_scores_folder {output_dir} \
            --retrieval_alpha {ra} --checkpoint checkpoints/{model}" 
        subprocess.run(cmd, shell=True)

for model in models:
    output_dir = f"results/{model}_nct{nct}_ncs{ncs}_{nr}"
    scores = []

    for seq in tqdm(df.DMS_id):
        if seq in problematics and 'mamba' in model:
            continue
        path = f"{output_dir}/{seq}.csv"

        if not os.path.exists(path):
            # print(model, seq, 'missing')
            continue

        df_ = pd.read_csv(path)
        scores.append(spearmanr(df_['DMS_score'],df_['predicted_score'])[0])

    mean = np.mean(scores)
    ci95 = 1.96 * np.std(scores) / np.sqrt(len(scores))

    print(f'{model}: {mean:.3f} +- {ci95:3f}')


