import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pos = np.arange(seq_len)[:,None]
    i = np.arange((d_model+1)//2)[None,:]
    div = np.power(base,(2*i)/d_model)
    angles = pos / div 
    pe = np.zeros((seq_len,d_model))
    pe[:,0::2] = np.sin(angles[:, :(d_model +1) // 2])[:, :len(pe[0,0::2])]
    pe[:, 1::2] = np.cos(angles[:, : d_model//2])[:, :len(pe[0, 1::2])]
    return pe 
    