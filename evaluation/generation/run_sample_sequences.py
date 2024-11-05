import subprocess
import json

model_name = "xlstm"
checkpoint = f"checkpoints/protxlstm_102M_60B"

parameters_list = [(10,1.,10,1.), (10,1.,15,1.), (10,1.,10,0.95), (10,0.9,10,0.95), (10,0.8,10,0.9),
                (100,1.,10,1.), (100,1.,15,1.), (100,1.,10,0.95), (100,0.9,10,0.95), (100,0.8,10,0.9),
                (500,1.,10,1.), (500,1.,15,1.), (500,1.,10,0.95), (500,0.9,10,0.95), (500,0.8,10,0.9),
                (1000,1.,10,1.), (1000,1.,15,1.), (1000,1.,10,0.95), (1000,0.9,10,0.95), (1000,0.8,10,0.9),
                (-1,1.,10,1.), (-1,1.,15,1.), (-1,1.,10,0.95), (-1,0.9,10,0.95), (-1,0.8,10,0.9)]

parameters_list = json.dumps(parameters_list)
                
for family_idx in [20, 22, 50, 98, 100, 141, 177, 222, 233, 265, 303, 327, 338, 341, 376, 393, 471, 479, 481]:

    family_idxs = json.dumps([family_idx])

    subprocess.run([
        'python', 'protxlstm/applications/sample_sequences.py',
        '--model_name', model_name,
        '--checkpoint', checkpoint,
        '--family_idxs', family_idxs,
        '--parameters_list', parameters_list,
        '--n_samples_per_family', str(100)])