import operator
from dataclasses import dataclass

import torch

import aine_drl.util as util
from aine_drl.exp import Action


@dataclass(frozen=True)
class DQNExperience:
    obs: torch.Tensor
    action: Action
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor

class DQNTrajectory:
    def __init__(
        self, 
        n_steps: int,
        sample_batch_size: int,
        capacity: int,
        num_envs: int,
        device: torch.device,
        online: bool = True
    ) -> None:
        self._training_freq = n_steps
        self._sample_batch_size = sample_batch_size
        self._capacity = capacity
        self._num_envs = num_envs
        self._device = device
        if not online:
            raise NotImplementedError("use only online learning. offline learning is not implemented yet.")
        self._online = online
        self.reset()
        
    def reset(self):
        """Reset experience replay, which is to clear memories."""
        self._n_steps = 0
        self._count = 0
        self._recent_idx = -1
        
        self._obs_buffer = self._make_buffer()
        self._action_buffer = self._make_buffer()
        self._reward_buffer = self._make_buffer()
        self._terminated_buffer = self._make_buffer()
        self._final_next_obs: list[torch.Tensor] = [None] * self._num_envs # type: ignore
        
    @property
    def can_sample(self) -> bool:
        """Is able to sample from experience replay. It depends on online/offline learning."""
        return self._count >= self._sample_batch_size and (not self._online or self._n_steps >= self._training_freq)
        
    def add(self, exp: DQNExperience):
        """Add an experience when you use online learning."""
        num_envs = exp.obs.shape[0]
        assert self._num_envs == num_envs
        
        self._n_steps += 1
        
        discrete_action = torch.split(exp.action.discrete_action, self._num_envs, dim=0)
        continuous_action = torch.split(exp.action.continuous_action, self._num_envs, dim=0)
        action = [Action(d, c) for d, c in zip(discrete_action, continuous_action)]
        
        for i in range(self._num_envs):
            self._recent_idx = (self._recent_idx + 1) % self._capacity
            self._count = min(self._count + 1, self._capacity)
            
            self._obs_buffer[self._recent_idx] = exp.obs[i]
            self._action_buffer[self._recent_idx] = action[i]
            self._reward_buffer[self._recent_idx] = exp.reward[i]
            self._terminated_buffer[self._recent_idx] = exp.terminated[i]
            self._final_next_obs[i] = exp.next_obs[i]
        
    def sample(self) -> DQNExperience:
        """Samples experience batch from it. Default sampling distribution is uniform."""
        self._n_steps = 0
        sample_idx = self._sample_idx()
        
        action_list = util.get_batch_list(self._action_buffer, sample_idx)
        action = Action.from_iter(action_list)
        
        return DQNExperience(
            DQNTrajectory._to_batch(self._obs_buffer, sample_idx),
            action,
            self._sample_next_obs(sample_idx),
            DQNTrajectory._to_batch(self._reward_buffer, sample_idx),
            DQNTrajectory._to_batch(self._terminated_buffer, sample_idx),
        )
    
    def _sample_idx(self) -> torch.Tensor:
        batch_idx = torch.randint(self._count, size=(self._sample_batch_size,), device=self._device)
        return batch_idx
        
    def _sample_next_obs(self, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Sample next obs from the trajectory.
        
        The source of this method is kengz/SLM-Lab (Github) https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/replay.py.

        Args:
            batch_idx (Tensor): batch indexes which mean current obs indexes

        Returns:
            Tensor: next obs batch
        """
        # [obs1, obs2, obs3, next_obs1, next_obs2, next_obs3]
        next_obs_idxs = (batch_idx + self._num_envs) % self._capacity
        # if recent < next_obs_index <= recent + num_envs, next_obs is stored in next_obs_buffer
        # it has two cases
        # case 1) [recent1, recent2, recent3, oldest1, oldest2, oldest3]
        # if batch_index is 1 (recent2), then 2 (recent=recent3) < 1+3 (next_obs_index=oldest2) <= 2+3
        # case 2) [prev1, prev2, prev3, recent1, recent2, recent3]
        # if batch_index is 4 (recent2), then 5 (recent=recent3) < 4+3 (next_obs_index not exists) < 5+3
        not_exsists_next_obs = torch.argwhere(
            (self._recent_idx < next_obs_idxs) & (next_obs_idxs <= self._recent_idx + self._num_envs)
        ).flatten()
        # check if there is any indexes to get from buffer
        do_replace = not_exsists_next_obs.size != 0
        if do_replace:
            # recent < next_obs_index <= recent + num_envs
            # i.e. 0 <= next_obs_index - recent - 1 < num_envs
            next_obs_buffer_idxs = next_obs_idxs[not_exsists_next_obs] - self._recent_idx - 1
            # to avoid index out of range exception due to the case 2
            next_obs_idxs[not_exsists_next_obs] = 0
        # get the next obs batch
        next_obs = DQNTrajectory._to_batch(self._obs_buffer, next_obs_idxs)
        if do_replace:
            # replace them
            next_obs[not_exsists_next_obs] = DQNTrajectory._to_batch(self._final_next_obs, next_obs_buffer_idxs)
        return next_obs

    def _make_buffer(self) -> list:
        return [None] * self._capacity
    
    @staticmethod
    def _to_batch(arr, idx) -> torch.Tensor:
        return torch.Tensor(operator.itemgetter(*idx)(arr))