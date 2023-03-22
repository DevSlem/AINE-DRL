from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from aine_drl.exp import Action, Experience
from aine_drl.trajectory.batch_trajectory import BatchTrajectory
from aine_drl.util import StaticRecursiveBuffer


@dataclass(frozen=True)
class PPOExperience:
    obs: torch.Tensor
    action: Action
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    state_value: torch.Tensor

class PPOTrajectory:
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
        
        self._final_next_obs = None
        
    def add(self, exp: PPOExperience):
        self._recent_idx += 1
        self._obs_buffer[self._recent_idx] = exp.obs
        self._action_buffer[self._recent_idx] = exp.action
        self._reward_buffer[self._recent_idx] = exp.reward
        self._terminated_buffer[self._recent_idx] = exp.terminated
        self._action_log_prob_buffer[self._recent_idx] = exp.action_log_prob
        self._state_value_buffer[self._recent_idx] = exp.state_value
        self._final_next_obs = exp.next_obs
        
    def sample(self) -> PPOExperience:
        self._obs_buffer.append(self._final_next_obs)
        exp_batch = PPOExperience(
            torch.cat(self._obs_buffer[:-1], dim=0),
            Action.from_iter(self._action_buffer),
            torch.cat(self._obs_buffer[1:], dim=0),
            torch.cat(self._reward_buffer, dim=0),
            torch.cat(self._terminated_buffer, dim=0),
            torch.cat(self._action_log_prob_buffer, dim=0),
            torch.cat(self._state_value_buffer, dim=0),
        )
        self.reset()
        return exp_batch
    
    def _make_buffer(self) -> list:
        return [None] * self._n_steps

@dataclass(frozen=True)
class RecurrentPPOExperience:
    obs: torch.Tensor
    action: Action
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    state_value: torch.Tensor
    hidden_state: torch.Tensor
    
class RecurrentPPOTrajectory:
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
        self._hidden_state_buffer = self._make_buffer()
        
        self._final_next_obs = None
        
    def add(self, exp: RecurrentPPOExperience):
        self._recent_idx += 1
        
        self._obs_buffer[self._recent_idx] = exp.obs
        self._action_buffer[self._recent_idx] = exp.action
        self._reward_buffer[self._recent_idx] = exp.reward
        self._terminated_buffer[self._recent_idx] = exp.terminated
        self._action_log_prob_buffer[self._recent_idx] = exp.action_log_prob
        self._state_value_buffer[self._recent_idx] = exp.state_value
        self._hidden_state_buffer[self._recent_idx] = exp.hidden_state
        self._final_next_obs = exp.next_obs
        
    def sample(self) -> RecurrentPPOExperience:
        self._obs_buffer.append(self._final_next_obs)
        exp_batch = RecurrentPPOExperience(
            torch.cat(self._obs_buffer[:-1], dim=0),
            Action.from_iter(self._action_buffer),
            torch.cat(self._obs_buffer[1:], dim=0),
            torch.cat(self._reward_buffer, dim=0),
            torch.cat(self._terminated_buffer, dim=0),
            torch.cat(self._action_log_prob_buffer, dim=0),
            torch.cat(self._state_value_buffer, dim=0),
            torch.cat(self._hidden_state_buffer, dim=0),
        )
        self.reset()
        return exp_batch
    
    def _make_buffer(self) -> list:
        return [None] * self._n_steps
    
class RecurrentPPORNDExperience(NamedTuple):
    obs: np.ndarray
    action: Action
    next_obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    int_reward: np.ndarray
    action_log_prob: np.ndarray
    ext_state_value: np.ndarray
    int_state_value: np.ndarray
    hidden_state: np.ndarray
    
class RecurrentPPORNDExperienceBatchTensor(NamedTuple):
    obs: torch.Tensor
    action: ActionTensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    int_reward: torch.Tensor
    action_log_prob: torch.Tensor
    ext_state_value: torch.Tensor
    int_state_value: torch.Tensor
    hidden_state: torch.Tensor
    n_steps: int

class RecurrentPPORNDTrajectory:
    def __init__(self, training_freq: int) -> None:
        fields = list(RecurrentPPORNDExperience._fields)
        fields.remove("next_obs")
        self._buffer = StaticRecursiveBuffer(tuple(fields), training_freq)
        
    @property
    def can_train(self) -> bool:
        return self._buffer.is_full
        
    def reset(self):
        self._next_obs_buffer = None
        self._buffer.reset()
        
    def add(self, experience: RecurrentPPORNDExperience):
        exp_dict = experience._asdict()
        self._next_obs_buffer = exp_dict.pop("next_obs")
        self._buffer.add(exp_dict)
    
    def sample(self, device: torch.device | None = None):
        exp_buffers = self._buffer.buffer_dict
        exp_batch = {}
        for key, buffer in exp_buffers.items():
            if key == "action":
                exp_batch[key] = Action.to_batch(buffer).to_action_tensor(device=device)
            elif key == "hidden_state":
                exp_batch[key] = torch.from_numpy(np.concatenate(buffer, axis=1)).to(device=device)
            else:
                exp_batch[key] = torch.from_numpy(np.concatenate(buffer, axis=0)).to(device=device)
        obs_buffer = exp_buffers["obs"]
        obs_buffer.append(self._next_obs_buffer)
        exp_batch["next_obs"] = torch.from_numpy(np.concatenate(obs_buffer[1:], axis=0)).to(device=device)
        exp_batch["n_steps"] = self._buffer.count
        self.reset()
        return RecurrentPPORNDExperienceBatchTensor(**exp_batch)