from __future__ import annotations
import datetime
import operator
import os
import random
from typing import Iterable, TypeVar

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.clip_grad import clip_grad_norm_

T = TypeVar("T")

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
    
def get_seed() -> int | None:
    global _random_seed
    return _random_seed

def get_items(arr: Iterable[T], idx: Iterable) -> tuple[T]:
    return operator.itemgetter(*idx)(tuple(arr))

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

def check_freq(value: int, frequency: int, constant_increment: int = 1) -> bool:
    """
    Check if the value is reached to the frequency. 
    It's analogous to (value % frequency == 0), 
    but it considers constant increment of the value.

    Args:
        constant_increment (int): constant increment of the value. Defaults to 1.
    """
    return value % frequency < constant_increment

def add_datetime_suffix(basename: str, delimiter: str = '_') -> str:
    """Add a datetime suffix wtih delimiter to the basename. (e.g. basename_220622_140322)"""
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    return delimiter.join([basename, suffix])

def exists_dir(directory) -> bool:
    return os.path.exists(directory)

def add_dir_num_suffix(basedir: str, start_num: int = 1, num_left: str = "", num_right: str = "") -> str:
    """Add a number suffix to the base directory. (e.g. basedir4)"""
    num = start_num
    dir = f"{basedir}{num_left}{num}{num_right}"
    while exists_dir(dir):
        num += 1
        dir = f"{basedir}{num_left}{num}{num_right}"
    return dir

def create_dir(directory):
    """If there's no directory, create it."""
    try:
        if not exists_dir(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def model_device(model: nn.Module) -> torch.device:
    """Returns the device of the model."""
    return next(model.parameters()).device

def batch2perenv(batch: torch.Tensor, num_envs: int) -> torch.Tensor:
    """
    `(num_envs x n_steps, *shape)` -> `(num_envs, n_steps, *shape)`
    
    The input `batch` must be like the following example `Before`:
    
    `num_envs` = 2, `n_steps` = 3
    
    Before::
    
        [env1_step0, 
         env2_step0, 
         env1_step1, 
         env2_step1, 
         env1_step2, 
         env2_step2]
         
    After::
    
        [[env1_step0, env1_step1, env1_step2],
         [env2_step0, env2_step1, env2_step2]]
    
    """
    shape = batch.shape
    # scalar data (num_envs * n,)
    if len(shape) < 2:
        return batch.reshape(-1, num_envs).T
    # non-scalar data (num_envs * n, *shape)
    else:
        shape = (-1, num_envs) + shape[1:]
        return batch.reshape(shape).transpose(0, 1)

def perenv2batch(per_env: torch.Tensor) -> torch.Tensor:
    """
    `(num_envs, n_steps, *shape)` -> `(num_envs x n_steps, *shape)`
    
    The input `per_env` must be like the following example `Before`:
    
    `num_envs` = 2, `n_steps` = 3
         
    Before::
    
        [[env1_step0, env1_step1, env1_step2],
         [env2_step0, env2_step1, env2_step2]]
         
    After::
    
        [env1_step0, 
         env2_step0, 
         env1_step1, 
         env2_step1, 
         env1_step2, 
         env2_step2]
    """
    shape = per_env.shape
    # scalar data (num_envs, n,)
    if len(shape) < 3:
        return per_env.T.reshape(-1)
    # non-scalar data (num_envs, n, *shape)
    else:
        shape = (-1,) + shape[2:]
        return per_env.transpose(0, 1).reshape(shape)
    
def copy_module(src_module: nn.Module, target_module: nn.Module):
    """
    Copy model weights from src to target.
    """
    target_module.load_state_dict(src_module.state_dict())

def polyak_update_module(src_module: nn.Module, target_module: nn.Module, src_ratio: float):
    assert src_ratio >= 0 and src_ratio <= 1
    for src_param, target_param in zip(src_module.parameters(), target_module.parameters()):
        target_param.data.copy_(src_ratio * src_param.data + (1.0 - src_ratio) * target_param.data)

