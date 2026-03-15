# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Separate positive (True label) and negative (False label) sequences 
    pos_seqs = [s for s, l in zip(seqs, labels) if l]
    neg_seqs = [s for s, l in zip(seqs, labels) if not l]

    # find size of the majority class, oversample minority class to match 
    max_count = max(len(pos_seqs), len(neg_seqs))

    # random sampling with replacement
    if len(pos_seqs) < max_count:
        # randomly select max_count indices from the pos sequences 
        pos_indices = np.random.choice(len(pos_seqs), size=max_count, replace=True)
        pos_seqs = [pos_seqs[i] for i in pos_indices]
    if len(neg_seqs) < max_count:
        # randomly select max_count indices from the neg sequences 
        neg_indices = np.random.choice(len(neg_seqs), size=max_count, replace=True)
        neg_seqs = [neg_seqs[i] for i in neg_indices]

    # combine positive and negative sets
    sampled_seqs = pos_seqs + neg_seqs
    sampled_labels = [True] * len(pos_seqs) + [False] * len(neg_seqs)

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    encoding_map = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
    }

    encodings = []
    for seq in seq_arr:
        encoded = []
        # look up one-hot vector for each nucleotide in the seq
        for nucleotide in seq.upper():
            encoded.extend(encoding_map.get(nucleotide, [0, 0, 0, 0]))
        encodings.append(encoded)

    return np.array(encodings)