from typing import NamedTuple
from aine_drl.trajectory.montecarlo_trajectory import MonteCarloTrajectory
from aine_drl.experience import ActionTensor, Experience
import torch

class REINFORCEExperienceBatch(NamedTuple):
    obs: torch.Tensor
    action: ActionTensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    entropy: torch.Tensor
    n_steps: int


class REINFORCETrajectory:
    def __init__(self) -> None:
        self.exp_trajectory = MonteCarloTrajectory()
        self.reset()
        
    @property
    def count(self) -> int:
        return self.exp_trajectory.count
        
    def reset(self):
        self.exp_trajectory.reset()
        
        self.action_log_prob = []
        self.entropy = []
        
    def add(self, 
            experience: Experience,
            action_log_prob: torch.Tensor,
            entropy: torch.Tensor):
        self.exp_trajectory.add(experience)
        self.action_log_prob.append(action_log_prob)
        self.entropy.append(entropy)
        
    def sample(self, device: torch.device | None = None) -> REINFORCEExperienceBatch:
        exp_batch = self.exp_trajectory.sample(device)
        exp_batch = REINFORCEExperienceBatch(
            exp_batch.obs,
            exp_batch.action,
            exp_batch.next_obs,
            exp_batch.reward,
            exp_batch.terminated,
            torch.cat(self.action_log_prob, dim=0).to(device=device),
            torch.cat(self.entropy, dim=0).to(device=device),
            exp_batch.n_steps
        )
        self.reset()
        return exp_batch
