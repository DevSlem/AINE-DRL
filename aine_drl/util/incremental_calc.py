from __future__ import annotations
from typing import Optional, Union, Tuple
import torch

class IncrementalMean:
    """
    Incremental mean calculation implementation. 
    It uses only two memories: mean, n.
    """
    def __init__(self) -> None:
        self.reset()
        
    def reset(self):
        self._mean = 0.0
        self._n = 0
        
    def update(self, value: float) -> float:
        """Update current average."""
        self._n += 1
        self._mean += (value - self._mean) / self._n
        return self._mean
        
    @property
    def mean(self) -> float:
        """Returns current average."""
        return self._mean
    
    @property
    def count(self) -> int:
        return self._n


class IncrementalMeanVarianceFromBatch:
    """
    ## Summary
    
    Incremental mean and variance calculation from batch. 
    See details in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.
    
    Args:
        unbiased (bool, optional): whether unbiased or biased variance. Defaults to unbiased variance.
        dim (int | None, optional): dimension along which mean and variance are computed. The default is to compute the values of the flattened array.
    """
    def __init__(
        self, 
        unbiased: bool = True, 
        dim: Optional[int] = None, 
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None
    ) -> None:
        self._unbiased = unbiased
        self._ddof = 1 if unbiased else 0
        self._dim = dim
        self._dtype = dtype
        self._device = device if device is not None else torch.device("cpu")
        self.reset()
        
    @property
    def mean(self) -> torch.Tensor:
        return self._mean
    
    @property
    def variance(self) -> torch.Tensor:
        return self._var
    
    @property
    def batch_size(self) -> int:
        return self._n
    
    def reset(self):
        """Reset the mean and variance to zero."""
        self._mean = torch.tensor(0.0, dtype=self._dtype, device=self._device)
        self._var = torch.tensor(0.0, dtype=self._dtype, device=self._device)
        self._n = 0
    
    def update(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the mean and variance from a batch of data.
        
        Returns:
            mean (float | np.ndarray): updated mean
            variance (float | np.ndarray): updated variance
        """
        if self._dim is None:
            batch_mean = batch.mean()
            batch_var = batch.var(unbiased=self._unbiased)
            batch_size = batch.numel()
        else:
            batch_mean = batch.mean(dim=self._dim)
            batch_var = batch.var(unbiased=self._unbiased, dim=self._dim)
            batch_size = batch.shape[self._dim]
        return self._update_from_batch_mean_var(batch_mean, batch_var, batch_size)
        
    def _update_from_batch_mean_var(self, 
                                    batch_mean: Union[float, torch.Tensor], 
                                    batch_var: Union[float, torch.Tensor], 
                                    batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # n: batch size
        # M: sum of squares
        # a: old batch
        # b: new batch
        # d: DDOF
        d = self._ddof
        
        n_a = self._n
        n_b = batch_size
        n = n_a + n_b
        
        # update mean
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * n_b / n

        # update variance
        M_a = self._var * (n_a - d)
        M_b = batch_var * (n_b - d)
        M = M_a + M_b + delta**2 * n_a * n_b / n
        self._var = M / (n - d)
        
        # update batch size
        self._n = n
        
        return self._mean, self._var
    