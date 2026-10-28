import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if len(seqs)==0 :
        return np.empty((0, 0), dtype=np.int64)
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    result = np.full((len(seqs), max_len), pad_value, dtype=np.int64)

    for i, seq in enumerate(seqs):
        trunc_seq = seq[:max_len]
        result[i, : len(trunc_seq)] = trunc_seq

    return result
            
        
    
            
    