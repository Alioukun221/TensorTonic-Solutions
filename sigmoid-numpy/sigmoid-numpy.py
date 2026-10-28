import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    array_list = np.asarray(x,dtype=float)
    return 1 / (1 + np.exp(-array_list))