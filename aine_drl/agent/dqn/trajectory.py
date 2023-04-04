import operator
from dataclasses import dataclass

import torch

import aine_drl.util.func as util_f
from aine_drl.exp import Action, Observation


@dataclass(frozen=True)
class DQNExperience:
    obs: Observation
    action: Action
    next_obs: Observation
    reward: torch.Tensor
    terminated: torch.Tensor
    
    def to(self, device: torch.device) -> "DQNExperience":
        return DQNExperience(
            self.obs.transform(lambda o: o.to(device)),
            self.action.transform(lambda a: a.to(device)),
            self.next_obs.transform(lambda o: o.to(device)),
            self.reward.to(device),
            self.terminated.to(device),
        )

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
        
        # the batch size of each item is 1
        self._obs_buffer: list[Observation] = self._make_buffer()
        self._action_buffer: list[Action] = self._make_buffer()
        self._reward_buffer: list[torch.Tensor] = self._make_buffer()
        self._terminated_buffer: list[torch.Tensor] = self._make_buffer()
        
        self._final_next_obs: list[Observation] = [None] * self._num_envs # type: ignore
        
    @property
    def can_sample(self) -> bool:
        """Is able to sample from experience replay. It depends on online/offline learning."""
        return self._count >= self._sample_batch_size and (not self._online or self._n_steps >= self._training_freq)
        
    def add(self, exp: DQNExperience):
        """Add an experience when you use online learning."""  
        self._n_steps += 1
        exp = exp.to(self._device)
        
        for i in range(self._num_envs):
            self._recent_idx = (self._recent_idx + 1) % self._capacity
            self._count = min(self._count + 1, self._capacity)
            
            self._obs_buffer[self._recent_idx] = exp.obs[i:i+1]
            self._action_buffer[self._recent_idx] = exp.action[i:i+1]
            self._reward_buffer[self._recent_idx] = exp.reward[i:i+1]
            self._terminated_buffer[self._recent_idx] = exp.terminated[i:i+1]
            self._final_next_obs[i] = exp.next_obs[i:i+1]
        
    def sample(self, device: torch.device) -> DQNExperience:
        """Samples experience batch from it. Default sampling distribution is uniform."""
        self._n_steps = 0
        sample_idx = self._sample_idx()
        
        obs = Observation.from_iter(util_f.get_items(self._obs_buffer, sample_idx))
        action = Action.from_iter(util_f.get_items(self._action_buffer, sample_idx))
        next_obs = self._sample_next_obs(sample_idx)
        reward = torch.cat(util_f.get_items(self._reward_buffer, sample_idx), dim=0)
        terminated = torch.cat(util_f.get_items(self._terminated_buffer, sample_idx), dim=0)
        
        return DQNExperience(obs, action, next_obs, reward, terminated).to(device=device)
    
    def _sample_idx(self) -> torch.Tensor:
        batch_idx = torch.randint(self._count, size=(self._sample_batch_size,))
        return batch_idx
        
    def _sample_next_obs(self, batch_idx: torch.Tensor) -> Observation:
        """
        Sample next obs from the trajectory.
        
        The source of this method is kengz/SLM-Lab (Github) https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/replay.py.
        """
        # num_envs = 3
        # [obs1, obs2, obs3, next_obs1, next_obs2, next_obs3]
        next_obs_idx = (batch_idx + self._num_envs) % self._capacity
        # if recent_idx < next_obs_idx <= recent_idx + num_envs, then next_obs is stored in final_next_obs
        # it has two cases
        # case 1) [recent1, recent2, recent3, oldest1, oldest2, oldest3]
        # if batch_idx is 1 (recent2), then 2 (recent_idx=recent3) < 1+3 (next_obs_idx=oldest2) <= 2+3
        # case 2) [prev1, prev2, prev3, recent1, recent2, recent3]
        # if batch_idx is 4 (recent2), then 5 (recent_idx=recent3) < 4+3 (next_obs_idx not exists) < 5+3
        not_exsists_next_obs = torch.argwhere(
            (self._recent_idx < next_obs_idx) & (next_obs_idx <= self._recent_idx + self._num_envs)
        ).flatten()
        # check if there is any indexes to get from buffer
        do_replace = len(not_exsists_next_obs) != 0
        if do_replace:
            # recent_idx < next_obs_idx <= recent_idx + num_envs
            # i.e. 0 <= next_obs_idx - recent_idx - 1 < num_envs
            final_next_obs_idx = next_obs_idx[not_exsists_next_obs] - self._recent_idx - 1
            # to avoid index out of range exception due to the case 2
            next_obs_idx[not_exsists_next_obs] = 0
        # get the next obs batch
        next_obs = Observation.from_iter(util_f.get_items(self._obs_buffer, next_obs_idx))
        if do_replace:
            # replace them
            next_obs[not_exsists_next_obs] = Observation.from_iter(util_f.get_items(self._final_next_obs, final_next_obs_idx)) # type: ignore
        return next_obs

    def _make_buffer(self) -> list:
        return [None] * self._capacity
    