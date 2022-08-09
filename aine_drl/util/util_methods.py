import operator
import numpy as np

def get_batch(arr, idxs) -> np.ndarray:
    """
    Returns a batch that consists of elements for the indexes from the array.

    Args:
        arr (Any): array
        idxs (Any): indexes

    Returns:
        np.ndarray: batch from the array
    """
    if isinstance(arr, np.ndarray):
        return arr[idxs]
    else:
        return np.array(operator.itemgetter(*idxs)(arr))