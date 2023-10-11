import numpy as np
from pyhmmer.easel import Alphabet
from pyhmmer.plan7 import HMM, Transitions
from scipy.special import softmax

from model_angelo.utils.residue_constants import restype_3_to_index, num_prot

pyhmmer_alphabet = {
    "amino": Alphabet.amino(),
    "DNA": Alphabet.dna(),
    "RNA": Alphabet.rna(),
}

alphabet_to_slice = {
    "amino": np.s_[..., :num_prot],
    "DNA": np.s_[..., num_prot : num_prot + 4],
    "RNA": np.s_[..., num_prot + 4 :],
    "PP": np.s_[..., num_prot:],
}


def convert_aa_logits_to_probs(
    aa_logits: np.ndarray,
    alphabet_type="amino",
) -> np.ndarray:
    """
    Process aa_logits to probabilities for a given alphabet.
    This includes just pyrimidine-purine matches for RNA.
    """
    processed_aa_logits = np.full_like(aa_logits, -100)
    processed_aa_logits[alphabet_to_slice[alphabet_type]] = aa_logits[
        alphabet_to_slice[alphabet_type]
    ]
    aa_probs = softmax(processed_aa_logits, axis=-1)
    if alphabet_type == "PP":
        alphabet_type = "RNA"  # Treat all purine-pyrimidine matches as RNA
        aa_probs_gather = np.zeros_like(aa_probs)
        aa_probs_gather[..., restype_3_to_index["G"]] = aa_probs[
            ...,
            [
                restype_3_to_index["DA"],
                restype_3_to_index["DG"],
                restype_3_to_index["A"],
                restype_3_to_index["G"],
            ],
        ].sum(axis=-1)
        aa_probs_gather[..., restype_3_to_index["C"]] = aa_probs_gather[
            ...,
            [
                restype_3_to_index["DC"],
                restype_3_to_index["DT"],
                restype_3_to_index["C"],
                restype_3_to_index["U"],
            ],
        ].sum(axis=-1)
        aa_probs = aa_probs_gather / aa_probs_gather.sum(axis=-1, keepdims=True)
    aa_probs = aa_probs[alphabet_to_slice[alphabet_type]]
    return aa_probs


def aa_logits_to_hmm(
    aa_logits: np.ndarray,
    confidence: np.ndarray = None,
    delta=0.05,
    gamma=0.5,
    alphabet_type="amino",
    name: str="model_angelo_search",
) -> HMM:
    """
    This function converts ModelAngelo log_probs to HMMER3 HMMs.
    The algorithm was developed with Lukas Kall and the code is based on
    https://github.com/althonos/pyhmmer/issues/40
    by Martin Larralde.
    """
    if confidence is None:
        confidence = np.ones(len(aa_logits))
    aa_probs = convert_aa_logits_to_probs(aa_logits, alphabet_type=alphabet_type)
    alphabet = pyhmmer_alphabet[alphabet_type]
    hmm = HMM(alphabet, M=len(aa_probs), name=bytes(f"{name}", "utf-8"))
    for res_index in range(len(aa_probs)):
        mm = max(confidence[res_index] - delta, gamma)
        for idx, p in enumerate(aa_probs[res_index]):
            hmm.match_emissions[res_index+1, idx] = p
            hmm.insert_emissions[res_index+1, idx] = p
        hmm.transition_probabilities[res_index+1, Transitions.MM] = mm
        if res_index < len(aa_probs) - 1:
            hmm.transition_probabilities[res_index+1, Transitions.MD] = hmm.transition_probabilities[res_index+1, Transitions.MI] = (1.0 - mm) / 2
            hmm.transition_probabilities[res_index+1, Transitions.IM] = hmm.transition_probabilities[res_index+1, Transitions.DM] = 1.0 - delta
            hmm.transition_probabilities[res_index+1, Transitions.II] = hmm.transition_probabilities[res_index+1, Transitions.DD] = delta
        else:
            # If res_index is the last, then TMD is 0 (Transitions to Match and Delete)
            hmm.transition_probabilities[res_index+1, Transitions.MI] = 1.0 - mm
            hmm.transition_probabilities[res_index+1, Transitions.IM] = 1.0 - delta
            hmm.transition_probabilities[res_index+1, Transitions.II] = delta
    hmm.set_composition()
    hmm.validate()
    return hmm


def dump_aa_logits_to_hmm_file(
    aa_logits: np.ndarray,
    output_file: str,
    confidence: np.ndarray = None,
    name: str = "model_angelo_search",
    alphabet_type: str = "amino",
):
    hmm = aa_logits_to_hmm(
        name=name, 
        aa_logits=aa_logits, 
        confidence=confidence, 
        alphabet_type=alphabet_type
    )
    with open(output_file, "wb") as f:
        hmm.write(f, binary=False)


if __name__ == "__main__":
    # For testing purposes
    probs = np.random.rand(100, 28) * 5
    probs /= probs.sum(axis=-1, keepdims=True)
    confidence = np.random.rand(100)
    hmm = aa_logits_to_hmm("test", probs, confidence, alphabet_type="RNA")
    print(hmm)
    with open("test.hmm", "wb") as f:
        hmm.write(f, binary=False)
