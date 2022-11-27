from typing import NamedTuple, Optional
from aine_drl.experience import Action, ActionTensor, Experience
import aine_drl.util as util
import numpy as np
import torch

class DoubleDQNExperienceBatch(NamedTuple):
    obs: torch.Tensor
    action: ActionTensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    n_steps: int


class DoubleDQNTrajectory:
    """
    Experience replay.
    """
    def __init__(self, 
                 training_freq: int,
                 batch_size: int,
                 capacity: int,
                 num_envs: int) -> None:
        self.training_freq = training_freq
        self.batch_size = batch_size
        self.capacity = capacity
        self.num_envs = num_envs
        self.reset()
        
    def reset(self):
        self.n_step = 0
        self._count = 0
        self._recent_idx = -1
        
        self.obs = [None] * self.capacity
        self.action = [None] * self.capacity
        self.reward = [None] * self.capacity
        self.terminated = [None] * self.capacity
        self.next_obs_buffer = [None] * self.num_envs
        
    @property
    def can_train(self) -> bool:
        return self._count >= self.batch_size and self.n_step >= self.training_freq
    
    @property
    def count(self) -> int:
        return self._count
    
    @property
    def recent_idx(self) -> int:
        return self._recent_idx
        
    def add(self, experience: Experience):
        num_envs = experience.obs.shape[0]
        assert self.num_envs == num_envs
        
        self.n_step += 1
        
        discrete_action = np.split(experience.action.discrete_action, num_envs, axis=0)
        continuous_action = np.split(experience.action.continuous_action, num_envs, axis=0)
        action = [Action.create(d, c) for d, c in zip(discrete_action, continuous_action)]
        
        for i in range(num_envs):
            self._recent_idx = (self._recent_idx + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)
            
            self.obs[self._recent_idx] = experience.obs[i]
            self.action[self._recent_idx] = action[i]
            self.reward[self._recent_idx] = experience.reward[i]
            self.terminated[self._recent_idx] = experience.terminated[i]
            self.next_obs_buffer[i] = experience.next_obs[i]
        
    def sample(self, device: Optional[torch.device] = None) -> DoubleDQNExperienceBatch:
        self.n_step = 0
        sample_idx = self._sample_idxs()
        
        actions = util.get_batch_list(self.action, sample_idx)
        action = Action.to_batch(actions).to_action_tensor(device)
        
        experience_batch = DoubleDQNExperienceBatch(
            self._get_batch_tensor(self.obs, sample_idx, device),
            action,
            torch.from_numpy(self._sample_next_obs(sample_idx)).to(device=device),
            self._get_batch_tensor(self.reward, sample_idx, device),
            self._get_batch_tensor(self.terminated, sample_idx, device),
            n_steps=self.batch_size
        )
        return experience_batch
    
    def _get_batch_tensor(self, items: list, batch_idx: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.from_numpy(util.get_batch(items, batch_idx)).to(device=device)
    
    def _sample_idxs(self) -> np.ndarray:
        batch_idxs = np.random.randint(self._count, size=self.batch_size)
        return batch_idxs
        
    def _sample_next_obs(self, batch_idxs: np.ndarray) -> np.ndarray:
        """
        Sample next obs from the trajectory. TODO: #6 It needs to be tested.
        
        The source of this method is kengz/SLM-Lab (Github) https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/replay.py.

        Args:
            batch_idxs (np.ndarray): batch indexes which mean current obs indexes

        Returns:
            np.ndarray: next obs batch
        """
        # [obs1, obs2, obs3, next_obs1, next_obs2, next_obs3]
        next_obs_idxs = (batch_idxs + self.num_envs) % self.capacity
        # if recent < next_obs_index <= recent + num_envs, next_obs is stored in next_obs_buffer
        # it has two cases
        # case 1) [recent1, recent2, recent3, oldest1, oldest2, oldest3]
        # if batch_index is 1 (recent2), then 2 (recent=recent3) < 1+3 (next_obs_index=oldest2) <= 2+3
        # case 2) [prev1, prev2, prev3, recent1, recent2, recent3]
        # if batch_index is 4 (recent2), then 5 (recent=recent3) < 4+3 (next_obs_index not exists) < 5+3
        not_exsists_next_obs = np.argwhere(
            (self.recent_idx < next_obs_idxs) & (next_obs_idxs <= self.recent_idx + self.num_envs)
        ).flatten()
        # check if there is any indexes to get from buffer
        do_replace = not_exsists_next_obs.size != 0
        if do_replace:
            # recent < next_obs_index <= recent + num_envs
            # i.e. 0 <= next_obs_index - recent - 1 < num_envs
            next_obs_buffer_idxs = next_obs_idxs[not_exsists_next_obs] - self.recent_idx - 1
            # to avoid index out of range exception due to the case 2
            next_obs_idxs[not_exsists_next_obs] = 0
        # get the next obs batch
        next_obs = util.get_batch(self.obs, next_obs_idxs)
        if do_replace:
            # replace them
            next_obs[not_exsists_next_obs] = util.get_batch(self.next_obs_buffer, next_obs_buffer_idxs)
        return next_obs
    