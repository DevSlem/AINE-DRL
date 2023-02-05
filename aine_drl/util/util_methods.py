import operator
from typing import Tuple, Union, Optional
import numpy as np
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.utils as torch_util
import torch.backends.cudnn as cudnn
import random

_random_seed = None

def seed(value: int):
    global _random_seed
    _random_seed = value
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(value)
    
def get_seed() -> Union[int, None]:
    global _random_seed
    return _random_seed

def get_batch_list(arr, idxs) -> list:
    return list(operator.itemgetter(*idxs)(arr))

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
    
def cumulative_average(data_list, data_count: int = 0) -> list:
    """ Returns a list containing averaged values of recent n data from the data list.

    Args:
        data_list (ArrayLike): data list
        data_count (int, optional): if it's less than 0, data_count is len(data_list). Defaults to 0.

    Returns:
        list: a list containing averaged values
    """
    
    data_list = np.array(data_list)
    averages = []
    if data_count <= 0:
        data_count = len(data_list)
        
    for i in range(len(data_list)):
        s = max(i - data_count + 1, 0)
        e = i + 1
        averages.append(np.mean(data_list[s:e]))
        
    return averages

def check_freq(value: int, frequency: int, constant_increment: int = 1) -> bool:
    """
    Check if the value is reached to the frequency. 
    It's analogous to (value % frequency == 0), 
    but it considers constant increment of the value.

    Args:
        constant_increment (int): constant increment of the value. Defaults to 1.
    """
    return value % frequency < constant_increment

def except_dict_element(dictionary: dict, key) -> dict:
    return {k: v for k, v in dictionary.items() if k != key}

def add_datetime_suffix(basename: str, delimiter: str = '_') -> str:
    """ Add a datetime suffix wtih delimiter to the basename. (e.g. basename_220622_140322) """
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    return delimiter.join([basename, suffix])

def exists_dir(directory) -> bool:
    return os.path.exists(directory)

def add_dir_num_suffix(basedir: str, start_num: int = 1, num_left: str = "", num_right: str = "") -> str:
    """ Add a number suffix to the base directory. (e.g. basedir4) """
    num = start_num
    dir = f"{basedir}{num_left}{num}{num_right}"
    while exists_dir(dir):
        num += 1
        dir = f"{basedir}{num_left}{num}{num_right}"
    return dir

def create_dir(directory):
    """ If there's no directory, create it. """
    try:
        if not exists_dir(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def get_model_device(model: nn.Module) -> torch.device:
    """Returns the device of the model."""
    return next(model.parameters()).device

def lr_scheduler_step(lr_scheduler, epoch: int):
    if lr_scheduler is not None:
        lr_scheduler.step(epoch=epoch)
    
def get_optim_params(optimizer: torch.optim.Optimizer):
    return [param["params"] for param in optimizer.param_groups]

def train_step(loss: torch.Tensor, 
               optimizer: torch.optim.Optimizer, 
               lr_scheduler = None, 
               grad_clip_max_norm: Union[float, None] = None,
               epoch: int = -1):
    optimizer.zero_grad()
    loss.backward()
    if grad_clip_max_norm is not None:
        torch_util.clip_grad_norm_(*get_optim_params(optimizer), grad_clip_max_norm)
    optimizer.step()
    lr_scheduler_step(lr_scheduler, epoch)
    
def total_training_steps(total_time_steps: int, training_freq: int, epoch: int = 1) -> int:
    return total_time_steps // training_freq * epoch

class IncrementalAverage:
    """
    Incremental average calculation implementation. 
    It uses only two memories: average, n.
    """
    def __init__(self) -> None:
        self.reset()
        
    def reset(self):
        self._average = 0.0
        self.n = 0
        
    def update(self, value):
        """Update current average."""
        self.n += 1
        self._average += (value - self._average) / self.n
        return self._average
        
    @property
    def average(self):
        """Returns current average."""
        return self._average
    
    @property
    def count(self) -> int:
        return self.n

class IncrementalMeanVarianceFromBatch:
    """
    ## Summary
    
    Incremental mean and variance calculation from batch. 
    See details in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.
    
    Args:
        ddof (int, optional): delta degrees of freedom (DDOF) - especially 0 means biased variance, 1 means unibased variance. Defaults to 1.
        axis (int | None, optional): axis along which mean and variance are computed. The default is to compute the values of the flattened array.
        
    ## Example
    
    default option with unbiased variance and along no axis::
    
        inc_mean_var = IncrementalMeanVarianceFromBatch()
        >>> inc_mean_var.update(np.array([2, 8, 7, 4, 5]))
        (5.2, 5.7)
        >>> inc_mean_var.update(np.array([-3, 5, 2, 6]))
        (4.0, 11.0)
        
    biased variance::
    
        inc_mean_var = IncrementalMeanVarianceFromBatch(ddof=0)
        >>> inc_mean_var.update(np.array([2, 8, 7, 4, 5]))
        (5.2, 4.5600000000000005)
        >>> inc_mean_var.update(np.array([-3, 5, 2, 6]))
        (4.0, 9.777777777777779)
        
    along axis 0 (it's useful when you want to compute online the mean and variance of each feature with many divided batches)::
    
        inc_mean_var = IncrementalMeanVarianceFromBatch(axis=0)
        >>> inc_mean_var.update(np.array([[2, 8, 7], 
                                          [-1, 10, 3],
                                          [-7, 18, 5]]))
        (array([-2., 12.,  5.]), array([21., 28.,  4.]))
        >>> inc_mean_var.update(np.array([[8, 2, 5],
                                          [-16, 4, 7]]))
        (array([-2.8,  8.4,  5.4]), array([83.7, 38.8,  2.8]))
    """
    def __init__(self, ddof: int = 1, axis: Optional[int] = None) -> None:
        self._ddof = ddof
        self._axis = axis
        self.reset()
        
    @property
    def mean(self) -> Union[float, np.ndarray]:
        return self._mean
    
    @property
    def variance(self) -> Union[float, np.ndarray]:
        return self._var
    
    @property
    def batch_size(self) -> int:
        return self._n
    
    def reset(self):
        """Reset the mean and variance to zero."""
        self._mean = 0.0
        self._var = 0.0
        self._n = 0
    
    def update(self, batch: np.ndarray) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Update the mean and variance from a batch of data.
        
        Returns:
            mean (float | np.ndarray): updated mean
            variance (float | np.ndarray): updated variance
        """
        batch_mean = batch.mean(axis=self._axis)
        batch_var = batch.var(ddof=self._ddof, axis=self._axis)
        batch_size = batch.size if self._axis is None else batch.shape[self._axis]
        return self._update_from_batch_mean_var(batch_mean, batch_var, batch_size)
        
    def _update_from_batch_mean_var(self, 
                                    batch_mean: Union[float, np.ndarray], 
                                    batch_var: Union[float, np.ndarray], 
                                    batch_size: int) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
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
        
    @property
    def state_dict(self) -> dict:
        return {
            "incremental_mean_variance_from_batch": {
                "mean": self._mean,
                "variance": self._var,
                "batch_size": self._n
            }
        }
        
    def load_state_dict(self, state_dict: dict):
        state_dict = state_dict["incremental_mean_variance_from_batch"]
        for key, value in state_dict.items():
            setattr(self, f"_{key}", value)