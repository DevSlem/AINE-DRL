from dataclasses import dataclass

import torch

from aine_drl.exp import Action, Observation


@dataclass(frozen=True)
class A2CExperience:
    obs: Observation
    action: Action
    next_obs: Observation
    reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    state_value: torch.Tensor
    entropy: torch.Tensor

class A2CTrajectory:
    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps
        self.reset()
        
    @property
    def reached_n_steps(self) -> bool:
        return self._recent_idx == self._n_steps - 1
        
    def reset(self):
        self._recent_idx = -1
        
        self._obs_buffer = self._make_buffer()
        self._action_buffer = self._make_buffer()
        self._reward_buffer = self._make_buffer()
        self._terminated_buffer = self._make_buffer()
        self._action_log_prob_buffer = self._make_buffer()
        self._state_value_buffer = self._make_buffer()
        self._entropy_buffer = self._make_buffer()
        
        self._final_next_obs = None
        
    def add(self, exp: A2CExperience):
        self._recent_idx += 1
        
        self._obs_buffer[self._recent_idx] = exp.obs
        self._action_buffer[self._recent_idx] = exp.action
        self._reward_buffer[self._recent_idx] = exp.reward
        self._terminated_buffer[self._recent_idx] = exp.terminated
        self._action_log_prob_buffer[self._recent_idx] = exp.action_log_prob
        self._state_value_buffer[self._recent_idx] = exp.state_value
        self._entropy_buffer[self._recent_idx] = exp.entropy
        self._final_next_obs = exp.next_obs
        
    def sample(self) -> A2CExperience:
        self._obs_buffer.append(self._final_next_obs)
        exp_batch = A2CExperience(
            Observation.from_iter(self._obs_buffer[:-1]),
            Action.from_iter(self._action_buffer),
            Observation.from_iter(self._obs_buffer[1:]),
            torch.cat(self._reward_buffer, dim=0),
            torch.cat(self._terminated_buffer, dim=0),
            torch.cat(self._action_log_prob_buffer, dim=0),
            torch.cat(self._state_value_buffer, dim=0),
            torch.cat(self._entropy_buffer, dim=0),
        )
        self.reset()
        return exp_batch

    def _make_buffer(self) -> list:
        return [None] * self._n_steps