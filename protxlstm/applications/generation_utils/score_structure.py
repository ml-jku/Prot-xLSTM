from Bio.PDB import PDBParser
import torch
from tqdm import tqdm
from transformers import EsmForProteinFolding

from protxlstm.utils import MASK_TO_ID


pdb_parser = PDBParser()


def compute_structure(seq, model):
    def keep_sequence(seq, l):
        if len(seq) > l:
            return False
        for mm in list(MASK_TO_ID.keys())+["<eos>", "<pad>", "<unk>", "<mask>", "<cls>", "<null_1>", "." , "-"]:
            if mm in seq:
                return False
        return True
    keep = keep_sequence(seq, l=750)
    if keep:
        with torch.no_grad():
            output = model.infer([seq])
        # pdb = model.output_to_pdb(output)
        ptm = output["ptm"].item()
        pae = output["predicted_aligned_error"].cpu().numpy()
        mean_plddt = ((output["plddt"] * output["atom37_atom_exists"]).sum(dim=(1, 2)) / output["atom37_atom_exists"].sum(dim=(1, 2))).item()
        pos_plddt = ((output["plddt"] * output["atom37_atom_exists"]).sum(dim=(2,)) / output["atom37_atom_exists"].sum(dim=(2,))).cpu().numpy()
    else:
        print(f"Sequence is invalid.")
        ptm, pae, mean_plddt, pos_plddt = 0, 0 ,0 , 0
    return ptm, pae, mean_plddt, pos_plddt


def score_structure(sequence_df, family_idx):

    assert len(set(list(sequence_df["family"]))) == 1 and sequence_df["family"].iloc[0] == family_idx

    device="cuda:0"

    # Import the folding model
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

    model = model.cuda(device)
    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    sequences = list(sequence_df["sequence"])
    for seq in tqdm(sequences):

        ptm, pae, mean_plddt, pos_plddt = compute_structure(seq, model)
        sequence_df.loc[sequence_df["sequence"] == seq, "ptm"] = ptm
        sequence_df.loc[sequence_df["sequence"] == seq, "mean_plddt"] = mean_plddt
        
    return sequence_df