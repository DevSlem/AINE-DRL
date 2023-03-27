from dataclasses import dataclass

import torch

from aine_drl.exp import Observation, Action


@dataclass(frozen=True)
class REINFORCEExperience:
    obs: Observation
    action: Action
    next_obs: Observation
    reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    entropy: torch.Tensor

class REINFORCETrajectory:
    def __init__(self) -> None:
        self.reset()
        
    @property
    def terminated(self) -> bool:
        return self._terminated
        
    def reset(self):
        self._terminated = False
        
        self._obs_buffer = []
        self._action_buffer = []
        self._reward_buffer = []
        self._terminated_buffer = []
        self._action_log_prob_buffer = []
        self._entropy_buffer = []
        
        self._fianl_next_obs = None
        
    def add(self, exp: REINFORCEExperience):
        self._terminated = (exp.terminated.item() > 0.5)
        
        self._obs_buffer.append(exp.obs)
        self._action_buffer.append(exp.action)
        self._reward_buffer.append(exp.reward)
        self._terminated_buffer.append(exp.terminated)
        self._action_log_prob_buffer.append(exp.action_log_prob)
        self._entropy_buffer.append(exp.entropy)
        self._fianl_next_obs = exp.next_obs
        
    def sample(self) -> REINFORCEExperience:
        self._obs_buffer.append(self._fianl_next_obs)
        exp_batch = REINFORCEExperience(
            Observation.from_iter(self._obs_buffer[:-1]),
            Action.from_iter(self._action_buffer),
            Observation.from_iter(self._obs_buffer[1:]),
            torch.cat(self._reward_buffer, dim=0),
            torch.cat(self._terminated_buffer, dim=0),
            torch.cat(self._action_log_prob_buffer, dim=0),
            torch.cat(self._entropy_buffer, dim=0),
        )
        self.reset()
        return exp_batch
