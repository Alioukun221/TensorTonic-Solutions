import numpy as np

def apply_mlm_mask(token_ids: np.ndarray, mask_positions: np.ndarray,
                   replace_probs: np.ndarray, random_tokens: np.ndarray,
                   mask_token_id: int = 103) -> dict:
    """
    Returns masked_ids and labels as int64 arrays in a dictionary.
    """
    labels = np.full_like(token_ids,-100)
    masked_ids = token_ids.copy()

    labels[mask_positions] = token_ids[mask_positions]

    mask_80 = mask_positions & (replace_probs < 0.8)
    masked_ids[mask_80] = mask_token_id

    # 10%: replace with random token
    mask_10_random = mask_positions & (replace_probs >= 0.8) & (replace_probs < 0.9)
    masked_ids[mask_10_random] = random_tokens[mask_10_random]

    # 10%: keep original token
    mask_10_unchanged = mask_positions & (replace_probs >= 0.9)
    masked_ids[mask_10_unchanged] = token_ids[mask_10_unchanged]

    return {
        "masked_ids": masked_ids,
        "labels": labels,
    }
    
    