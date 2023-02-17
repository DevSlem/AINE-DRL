from aine_drl.experience import Action, Experience, ExperienceBatchTensor
import torch
import numpy as np

class MonteCarloTrajectory:
    """
    It's a trajectory utility of experience batch for the monte carlo (MC) method. 
    It works to only one environment.
    """
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self._recent_idx = -1
        self.obs = []
        self.action = []
        self.reward = []
        self.terminated = []
        self.next_obs_buffer = None
        
    @property
    def count(self) -> int:
        return self._recent_idx + 1
    
    @property
    def recent_idx(self) -> int:
        return self._recent_idx
    
    def add(self, experience: Experience):
        self._recent_idx += 1
        
        self.obs.append(experience.obs)
        self.action.append(experience.action)
        self.reward.append(experience.reward)
        self.terminated.append(experience.terminated)
        self.next_obs_buffer = experience.next_obs
        
    def sample(self, device: torch.device | None = None) -> ExperienceBatchTensor:
        self.obs.append(self.next_obs_buffer)
        exp_batch = ExperienceBatchTensor(
            torch.from_numpy(np.concatenate(self.obs[:-1], axis=0)).to(device=device),
            Action.to_batch(self.action).to_action_tensor(device),
            torch.from_numpy(np.concatenate(self.obs[1:], axis=0)).to(device=device),
            torch.from_numpy(np.concatenate(self.reward, axis=0)).to(device=device),
            torch.from_numpy(np.concatenate(self.terminated, axis=0)).to(device=device),
            self.count
        )
        return exp_batch
