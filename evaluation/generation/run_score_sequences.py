import subprocess

model_name = "protxlstm_102M_60B"
        
for family_idx in [20, 22, 50, 98, 100, 141, 177, 222, 233, 265, 303, 327, 338, 341, 376, 393, 471, 479, 481]:

    subprocess.run([
        'python', 'protxlstm/applications/score_sequences.py',
        '--model_name', model_name,
        '--family_idx', str(family_idx),
        '--num_sequences', str(100)])