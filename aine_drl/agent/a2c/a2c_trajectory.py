from typing import NamedTuple
from aine_drl.trajectory.batch_trajectory import BatchTrajectory
from aine_drl.experience import ActionTensor, Experience
import torch

class A2CExperienceBatch(NamedTuple):
    obs: torch.Tensor
    action: ActionTensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    v_pred: torch.Tensor
    entropy: torch.Tensor
    n_steps: int


class A2CTrajectory:
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
        self.entropy = [None] * self.max_n_steps
        
    def add(self, 
            experience: Experience,
            action_log_prob: torch.Tensor,
            v_pred: torch.Tensor,
            entropy: torch.Tensor):
        self.exp_trajectory.add(experience)
        self.action_log_prob[self.exp_trajectory.recent_idx] = action_log_prob
        self.v_pred[self.exp_trajectory.recent_idx] = v_pred
        self.entropy[self.exp_trajectory.recent_idx] = entropy
        
    def sample(self, device: torch.device | None = None) -> A2CExperienceBatch:
        exp_batch = self.exp_trajectory.sample(device)
        exp_batch = A2CExperienceBatch(
            exp_batch.obs,
            exp_batch.action,
            exp_batch.next_obs,
            exp_batch.reward,
            exp_batch.terminated,
            torch.cat(self.action_log_prob, dim=0).to(device=device),
            torch.cat(self.v_pred, dim=0).to(device=device),
            torch.cat(self.entropy, dim=0).to(device=device),
            exp_batch.n_steps
        )
        self.reset()
        return exp_batch
