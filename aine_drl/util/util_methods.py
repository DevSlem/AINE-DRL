import operator
import numpy as np

def get_batch(arr, idxs: np.ndarray) -> np.ndarray:
    """Returns a batch that consists of elements for the indexes from the array.

    Args:
        arr (ArrayLike): array
        idxs (np.ndarray): indexes

    Returns:
        np.ndarray: batch from the array
    """
    if isinstance(arr, np.ndarray):
        return np.array(operator.itemgetter(*idxs)(arr))
    else:
        return arr[idxs]