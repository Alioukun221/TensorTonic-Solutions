import numpy as np

def bert_embeddings(token_ids: np.ndarray, segment_ids: np.ndarray,
                    token_embeddings: np.ndarray, position_embeddings: np.ndarray,
                    segment_embeddings: np.ndarray) -> np.ndarray:
    """
    Returns the float64 BERT input embeddings with shape (B, S, H).
    """
    token_embeddings = token_embeddings[token_ids]
    segment_embeddings = segment_embeddings[segment_ids]

    sequence_position = np.arange(token_ids.shape[1])
    position_embeddings = position_embeddings[sequence_position]

    embeddings = token_embeddings + position_embeddings + segment_embeddings 

    return embeddings.astype("float64")