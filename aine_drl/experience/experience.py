from __future__ import annotations
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
    
    @staticmethod
    def create(states, actions, next_states, rewards, terminateds) -> ExperienceBatch:
        """
        Helper method to create an ExperienceBatch instance. 
        This method converts each argument into numpy array.
        """
        experience_batch = ExperienceBatch(
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards),
            np.array(terminateds)
        )
        return experience_batch
