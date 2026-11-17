import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """
    return gamma * ((x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(np.var(x, axis=-1, keepdims=True) + eps)) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
    """
    is_2d = Q.ndim ==2 
    if is_2d :
         Q, K, V = Q[None, ...], K[None, ...], V[None, ...]

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    batch_size,n_q,d_model = Q_proj.shape 
    n_k = K_proj.shape[1]
    n_v = V_proj.shape[1]
    d_k = d_model // num_heads 

    Q_heads = Q_proj.reshape(batch_size, n_q , num_heads , d_k).transpose(0,2,1,3)
    K_heads = K_proj.reshape(batch_size, n_k , num_heads , d_k).transpose(0,2,1,3)
    V_heads = V_proj.reshape(batch_size, n_v, num_heads , d_k).transpose(0,2,1,3)

    weights = softmax((Q_heads @ K_heads.swapaxes(-1,-2)) / np.sqrt(d_k))

    head_output = weights @ V_heads 
    concat = head_output.transpose(0,2,1,3).reshape(batch_size, n_q , d_model)
    output = concat @ W_o

    return output[0] if is_2d else output
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    return np.maximum(0, x @ W1 + b1) @ W2 + b2 

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    z = layer_norm(x + multi_head_attention(x ,x ,x ,W_q ,W_k ,W_v ,W_o ,num_heads), gamma1, beta1)
    y = layer_norm(z + feed_forward(z, W1, b1, W2, b2), gamma2, beta2)

    return y 
