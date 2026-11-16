import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    is_2d = (Q.ndim == 2)
    if is_2d:
        Q, K, V = Q[None, ...], K[None, ...], V[None, ...]

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    batch_size, n_q, d_model = Q_proj.shape
    n_k = K_proj.shape[1]
    n_v = V_proj.shape[1]
    d_k = d_model // num_heads

    # (batch_size, num_heads, seq_len, d_k)
    Q_heads = Q_proj.reshape(batch_size, n_q, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, n_k, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, n_v, num_heads, d_k).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scores = (Q_heads @ K_heads.swapaxes(-1, -2)) / np.sqrt(d_k)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # (batch_size, num_heads, n_q, d_k) -> (batch_size, n_q, d_model)
    head_output = weights @ V_heads
    concat = head_output.transpose(0, 2, 1, 3).reshape(batch_size, n_q, d_model)

    output = concat @ W_o

    return output[0] if is_2d else output