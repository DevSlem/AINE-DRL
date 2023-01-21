from typing import NamedTuple, Optional
from aine_drl.trajectory.batch_trajectory import BatchTrajectory
from aine_drl.experience import ActionTensor, Experience, Action
import torch
from aine_drl.util import StaticRecursiveBuffer
import numpy as np

class PPOExperienceBatch(NamedTuple):
    obs: torch.Tensor
    action: ActionTensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    v_pred: torch.Tensor
    n_steps: int

class PPOTrajectory:
    def __init__(self, max_n_steps: int) -> None:
        self.max_n_steps = max_n_steps
        self.exp_trajectory = BatchTrajectory(max_n_steps)
        self.reset()
        
    @property
    def count(self) -> int:
        return self.exp_trajectory.count
        
    def reset(self):
        self.exp_trajectory.reset()
        
        self.action_log_prob = [None] * self.max_n_steps
        self.v_pred = [None] * self.max_n_steps
        
    def add(self, 
            experience: Experience,
            action_log_prob: torch.Tensor,
            v_pred: torch.Tensor):
        self.exp_trajectory.add(experience)
        self.action_log_prob[self.exp_trajectory.recent_idx] = action_log_prob
        self.v_pred[self.exp_trajectory.recent_idx] = v_pred
        
    def sample(self, device: Optional[torch.device] = None) -> PPOExperienceBatch:
        exp_batch = self.exp_trajectory.sample(device)
        exp_batch = PPOExperienceBatch(
            exp_batch.obs,
            exp_batch.action,
            exp_batch.next_obs,
            exp_batch.reward,
            exp_batch.terminated,
            torch.cat(self.action_log_prob, dim=0).to(device=device),
            torch.cat(self.v_pred, dim=0).to(device=device),
            exp_batch.n_steps
        )
        self.reset()
        return exp_batch

class RecurrentPPOExperienceBatch(NamedTuple):
    obs: torch.Tensor
    action: ActionTensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    v_pred: torch.Tensor
    hidden_state: torch.Tensor
    n_steps: int
    
class RecurrentPPOTrajectory:
    def __init__(self, max_n_steps: int) -> None:
        self.max_n_steps = max_n_steps
        self.exp_trajectory = BatchTrajectory(max_n_steps)
        self.reset()
        
    @property
    def count(self) -> int:
        return self.exp_trajectory.count
    
    def reset(self):
        self.exp_trajectory.reset()
        
        self.action_log_prob = [None] * self.max_n_steps
        self.v_pred = [None] * self.max_n_steps
        self.hidden_state = [None] * self.max_n_steps
        
    def add(self, 
            experience: Experience,
            action_log_prob: torch.Tensor,
            v_pred: torch.Tensor,
            hidden_state: torch.Tensor):
        self.exp_trajectory.add(experience)
        self.action_log_prob[self.exp_trajectory.recent_idx] = action_log_prob
        self.v_pred[self.exp_trajectory.recent_idx] = v_pred
        self.hidden_state[self.exp_trajectory.recent_idx] = hidden_state
        
    def sample(self, device: Optional[torch.device] = None) -> RecurrentPPOExperienceBatch:
        exp_batch = self.exp_trajectory.sample(device)
        exp_batch = RecurrentPPOExperienceBatch(
            exp_batch.obs,
            exp_batch.action,
            exp_batch.next_obs,
            exp_batch.reward,
            exp_batch.terminated,
            torch.cat(self.action_log_prob, dim=0).to(device=device),
            torch.cat(self.v_pred, dim=0).to(device=device),
            torch.cat(self.hidden_state, dim=1).to(device=device),
            exp_batch.n_steps
        )
        self.reset()
        return exp_batch
    
class RecurrentPPORNDExperience(NamedTuple):
    obs: np.ndarray
    action: Action
    next_obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
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
    
    def sample(self, device: Optional[torch.device] = None):
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