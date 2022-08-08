import numpy as np
from typing import NamedTuple

class Experience(NamedTuple):
    """
    To store an experience for an agent in one time step. 
    """
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float
    terminated: bool
    
class ExperienceBatch(NamedTuple):
    """
    It's a batch of experiences. The first dimension of every field indicates batch size. 
    """
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    terminateds: np.ndarray
