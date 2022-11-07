from typing import Optional
from aine_drl.experience import Action, Experience, ExperienceBatchTensor
import torch
import numpy as np

class BatchTrajectory:
    """
    It's a trajectory utility of experience batch for the batch learning.
    
    Args:
        max_n_steps (int): maximum number of time steps to be stored
    """
    def __init__(self, max_n_steps: int) -> None:
        self.max_n_steps = max_n_steps
        self.reset()
        
    @property
    def count(self) -> int:
        return self._count
    
    @property
    def recent_idx(self) -> int:
        return self._recent_idx
        
    def reset(self):
        self._count = 0
        self._recent_idx = -1
        
        self.obs = [None] * self.max_n_steps
        self.action = [None] * self.max_n_steps
        self.reward = [None] * self.max_n_steps
        self.terminated = [None] * self.max_n_steps
        self.next_obs_buffer = None # most recently added next state
        
    def add(self, experience: Experience):
        self._recent_idx = (self._recent_idx + 1) % self.max_n_steps
        self._count = min(self._count + 1, self.max_n_steps)
        
        self.obs[self._recent_idx] = experience.obs
        self.action[self._recent_idx] = experience.action
        self.reward[self._recent_idx] = experience.reward
        self.terminated[self._recent_idx] = experience.terminated
        self.next_obs_buffer = experience.next_obs
    
    def sample(self, device: Optional[torch.device] = None) -> ExperienceBatchTensor:
        self.obs.append(self.next_obs_buffer)
        exp_batch = ExperienceBatchTensor(
            torch.from_numpy(np.concatenate(self.obs[:-1], axis=0)).to(device=device),
            Action.to_batch(self.action).to_action_tensor(device),
            torch.from_numpy(np.concatenate(self.obs[1:], axis=0)).to(device=device),
            torch.from_numpy(np.concatenate(self.reward, axis=0)).to(device=device),
            torch.from_numpy(np.concatenate(self.terminated, axis=0)).to(device=device),
            self.count
        )
        return exp_batch