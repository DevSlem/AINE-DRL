from dataclasses import dataclass

import torch

from aine_drl.exp import Action, Observation


@dataclass(frozen=True)
class PPOExperience:
    obs: Observation
    action: Action
    next_obs: Observation
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
            Observation.from_iter(self._obs_buffer[:-1]),
            Action.from_iter(self._action_buffer),
            Observation.from_iter(self._obs_buffer[1:]),
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
    obs: Observation
    action: Action
    next_obs: Observation
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
            Observation.from_iter(self._obs_buffer[:-1]),
            Action.from_iter(self._action_buffer),
            Observation.from_iter(self._obs_buffer[1:]),
            torch.cat(self._reward_buffer, dim=0),
            torch.cat(self._terminated_buffer, dim=0),
            torch.cat(self._action_log_prob_buffer, dim=0),
            torch.cat(self._state_value_buffer, dim=0),
            torch.cat(self._hidden_state_buffer, dim=1),
        )
        self.reset()
        return exp_batch
    
    def _make_buffer(self) -> list:
        return [None] * self._n_steps
    
@dataclass(frozen=True)
class PPORNDExperience:
    obs: Observation
    action: Action
    next_obs: Observation
    ext_reward: torch.Tensor
    int_reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    ext_state_value: torch.Tensor
    int_state_value: torch.Tensor
    
class PPORNDTrajectory:
    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps
        self.reset()
        
    @property
    def reached_n_steps(self) -> int:
        return self._recent_idx == self._n_steps - 1
    
    def reset(self):
        self._recent_idx = -1
        
        self._obs_buffer = self._make_buffer()
        self._action_buffer = self._make_buffer()
        self._ext_reward_buffer = self._make_buffer()
        self._int_reward_buffer = self._make_buffer()
        self._terminated_buffer = self._make_buffer()
        self._action_log_prob_buffer = self._make_buffer()
        self._ext_state_value_buffer = self._make_buffer()
        self._int_state_value_buffer = self._make_buffer()
        
        self._final_next_obs = None
        
    def add(self, exp: PPORNDExperience):
        self._recent_idx += 1
        
        self._obs_buffer[self._recent_idx] = exp.obs
        self._action_buffer[self._recent_idx] = exp.action
        self._ext_reward_buffer[self._recent_idx] = exp.ext_reward
        self._int_reward_buffer[self._recent_idx] = exp.int_reward
        self._terminated_buffer[self._recent_idx] = exp.terminated
        self._action_log_prob_buffer[self._recent_idx] = exp.action_log_prob
        self._ext_state_value_buffer[self._recent_idx] = exp.ext_state_value
        self._int_state_value_buffer[self._recent_idx] = exp.int_state_value
        self._final_next_obs = exp.next_obs
        
    def sample(self) -> PPORNDExperience:
        self._obs_buffer.append(self._final_next_obs)
        exp_batch = PPORNDExperience(
            Observation.from_iter(self._obs_buffer[:-1]),
            Action.from_iter(self._action_buffer),
            Observation.from_iter(self._obs_buffer[1:]),
            torch.cat(self._ext_reward_buffer, dim=0),
            torch.cat(self._int_reward_buffer, dim=0),
            torch.cat(self._terminated_buffer, dim=0),
            torch.cat(self._action_log_prob_buffer, dim=0),
            torch.cat(self._ext_state_value_buffer, dim=0),
            torch.cat(self._int_state_value_buffer, dim=0),
        )
        self.reset()
        return exp_batch
        
    def _make_buffer(self) -> list:
        return [None] * self._n_steps
    
@dataclass(frozen=True)
class RecurrentPPORNDExperience:
    obs: Observation
    action: Action
    next_obs: Observation
    ext_reward: torch.Tensor
    int_reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    ext_state_value: torch.Tensor
    int_state_value: torch.Tensor
    hidden_state: torch.Tensor
    next_hidden_state: torch.Tensor

class RecurrentPPORNDTrajectory:
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
        self._ext_reward_buffer = self._make_buffer()
        self._int_reward_buffer = self._make_buffer()
        self._terminated_buffer = self._make_buffer()
        self._action_log_prob_buffer = self._make_buffer()
        self._ext_state_value_buffer = self._make_buffer()
        self._int_state_value_buffer = self._make_buffer()
        self._hidden_state_buffer = self._make_buffer()
        
        self._final_next_obs = None
        self._final_next_hidden_state = None
        
    def add(self, exp: RecurrentPPORNDExperience):
        self._recent_idx += 1
        
        self._obs_buffer[self._recent_idx] = exp.obs
        self._action_buffer[self._recent_idx] = exp.action
        self._ext_reward_buffer[self._recent_idx] = exp.ext_reward
        self._int_reward_buffer[self._recent_idx] = exp.int_reward
        self._terminated_buffer[self._recent_idx] = exp.terminated
        self._action_log_prob_buffer[self._recent_idx] = exp.action_log_prob
        self._ext_state_value_buffer[self._recent_idx] = exp.ext_state_value
        self._int_state_value_buffer[self._recent_idx] = exp.int_state_value
        self._hidden_state_buffer[self._recent_idx] = exp.hidden_state
        self._final_next_obs = exp.next_obs
        self._final_next_hidden_state = exp.next_hidden_state
    
    def sample(self) -> RecurrentPPORNDExperience:
        self._obs_buffer.append(self._final_next_obs)
        self._hidden_state_buffer.append(self._final_next_hidden_state)
        exp_batch = RecurrentPPORNDExperience(
            Observation.from_iter(self._obs_buffer[:-1]),
            Action.from_iter(self._action_buffer),
            Observation.from_iter(self._obs_buffer[1:]),
            torch.cat(self._ext_reward_buffer, dim=0),
            torch.cat(self._int_reward_buffer, dim=0),
            torch.cat(self._terminated_buffer, dim=0),
            torch.cat(self._action_log_prob_buffer, dim=0),
            torch.cat(self._ext_state_value_buffer, dim=0),
            torch.cat(self._int_state_value_buffer, dim=0),
            torch.cat(self._hidden_state_buffer[:-1], dim=1),
            torch.cat(self._hidden_state_buffer[1:], dim=1),
        )
        self.reset()
        return exp_batch
    
    def _make_buffer(self) -> list:
        return [None] * self._n_steps