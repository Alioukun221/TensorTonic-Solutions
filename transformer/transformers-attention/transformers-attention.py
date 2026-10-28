import torch
import math 
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    d_k = K.shape[-1]

    score_scaled = (Q @ K.transpose(-2,-1))/ math.sqrt(d_k)
    attention_weight = torch.softmax(score_scaled,dim= -1)

    return attention_weight @ V 
    