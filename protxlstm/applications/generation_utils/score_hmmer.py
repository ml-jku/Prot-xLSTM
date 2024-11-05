import string
from Bio import SeqIO
import pyhmmer
from tqdm import tqdm

alphabet = pyhmmer.easel.Alphabet.amino()

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str):
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

def read_msa_unaligned(filename: str):
    """ Reads the sequences from an MSA file, removes only . - and * characters."""
    return [(record.description, str(record.seq).replace(".","").replace("-","").replace("*","").upper()) for record in SeqIO.parse(filename, "fasta")]

def check_msa(msa):
    """ Checks if there are any repeated sequences in the MSA"""
    seqs = set()
    for el in msa:
        seqs.add(el[1])
    assert len(seqs) == len(msa), "There are repeated sequences in the MSA"
    
def make_hmm_from_a3m_msa(msa_filepath, hmm_filename=None):
    # Load MSA from a3m
    msa_tup = read_msa(msa_filepath)
    # check_msa(msa_tup)
    # Create digitized MSA block
    all_seqs = [pyhmmer.easel.TextSequence(name=str(i).encode("utf-8"), sequence=seq) for i, (idz, seq) in enumerate(msa_tup)]
    msa  = pyhmmer.easel.TextMSA(name=b"msa", sequences=all_seqs)
    msa = msa.digitize(alphabet)
    # Fit HMM
    builder = pyhmmer.plan7.Builder(alphabet)
    background = pyhmmer.plan7.Background(alphabet)
    hmm, _, _ = builder.build_msa(msa, background)
    if hmm_filename is not None:
        with open(f"{hmm_filename}.hmm", "wb") as output_file:
            hmm.write(output_file)
    return hmm

def align_and_score_sequences_in_a3m_with_hmm(hmm, sequences_path=None, sequences_list=None):
    if sequences_list is not None:
        msa = sequences_list
        all_seqs = [pyhmmer.easel.TextSequence(name=str(i).encode("utf-8"), sequence=seq) for i, seq in enumerate(sequences_list)]
    elif sequences_path is not None:
        # Load sequences from a3m
        msa = read_msa_unaligned(sequences_path)
        all_seqs = [pyhmmer.easel.TextSequence(name=str(i).encode("utf-8"), sequence=seq) for i, (idz, seq) in enumerate(msa)]
    else:
        raise NotImplementedError("Missing sequences to align/score")
    # Create digitized Sequence block
    seq_block = pyhmmer.easel.TextSequenceBlock(all_seqs)
    seq_block = seq_block.digitize(alphabet)
    # Get all hits from the hmm
    background = pyhmmer.plan7.Background(alphabet)
    pipeline = pyhmmer.plan7.Pipeline(alphabet, background=background, bias_filter=False, F1=1.0, F2=1.0, F3=1.0)
    hits = pipeline.search_hmm(hmm, seq_block)
    if len(hits) != len(msa):
        print(f"Number of hits: {len(hits)} is different from the number of sequences in the MSA: {len(msa)}")
    # Extract hits
    all_hits = {}
    for hit in hits:
        idz, score, evalue = hit.name, hit.score, hit.evalue
        i = int(idz.decode("utf-8"))
        seq = msa[i][1] if sequences_path is not None else sequences_list[i]
        all_hits[seq] = {"score": score, "evalue": evalue}
    return all_hits


def score_hmmer(sequence_df, family_idx, data_dir = f"./data/"):

    assert len(set(list(sequence_df["family"]))) == 1 and sequence_df["family"].iloc[0] == family_idx
    
    family_id = sequence_df["family_id"].iloc[0]
    msa_filepath = f"{data_dir}/a3m_files/{family_id}/a3m/uniclust30.a3m"
    try:
        hmm = make_hmm_from_a3m_msa(msa_filepath)
    except:
        raise Exception(f"Missing MSA of family {family_id}")
    
    # align sequences
    sequences = list(sequence_df["sequence"])
    scores = align_and_score_sequences_in_a3m_with_hmm(hmm, sequences_list=sequences)

    # save the scores associated to each sequence in the main df in the columns "score" and "evalue"
    for seq in tqdm(sequences):
        sequence_df.loc[sequence_df["sequence"] == seq, "score_gen"] = scores[seq]["score"] if seq in scores.keys() else 0
        sequence_df.loc[sequence_df["sequence"] == seq, "evalue_gen"] = scores[seq]["evalue"] if seq in scores.keys() else 1

    return sequence_df


            