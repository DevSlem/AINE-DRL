import operator
from typing import Union
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
    return next(model.parameters()).device

def lr_scheduler_step(lr_scheduler, epoch: int):
    if lr_scheduler is not None:
        lr_scheduler.step(epoch=epoch)

def vector_env_pack(batch_tensor: torch.Tensor, num_envs: int) -> torch.Tensor:
    """
    Pack a batch tensor from (batch_size * num_envs, *shape) to (batch_size, num_envs, *shape).
    Reference: https://github.com/kengz/SLM-Lab/blob/9102ff923d7a3e9c579edc18c6547cce94a7b77a/slm_lab/lib/math_util.py#L35
    """
    shape = list(batch_tensor.shape)
    if len(shape) < 2: # scalara data (batch_size * num_envs,)
        return batch_tensor.view(-1, num_envs)
    else: # non-scalar data (batch_size * num_envs, *shape)
        pack_shape = [-1, num_envs] + shape[1:]
        return batch_tensor.view(pack_shape)
    
def vector_env_unpack(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    Unpack a batch tensor from (batch_size, num_envs, *shape) to (batch_size * num_envs, *shape).
    Reference: https://github.com/kengz/SLM-Lab/blob/9102ff923d7a3e9c579edc18c6547cce94a7b77a/slm_lab/lib/math_util.py#L45
    """
    shape = list(batch_tensor.shape)
    if len(shape) < 3: # scalar data (batch_size, num_envs,)
        return batch_tensor.view(-1)
    else: # non-scalar data (batch_size, num_envs, *shape) 
        unpack_shape = [-1] + shape[2:]
        return batch_tensor.view(unpack_shape)
    
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
