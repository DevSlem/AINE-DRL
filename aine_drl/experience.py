from typing import NamedTuple, List, Optional
import numpy as np
import torch

class Action(NamedTuple):
    """
    Standard action data type with numpy array. `batch_size` must be `num_envs` x `n_steps`.

    Args:
        discrete_action (ndarray): shape must be `(batch_size, num_discrete_branches)`
        continuous_action (ndarray): shape must be `(batch_size, num_continuous_branches)`
    """
    
    discrete_action: np.ndarray
    continuous_action: np.ndarray
    
    @property
    def num_discrete_branches(self) -> int:
        """Number of discrete action branches."""
        return self.discrete_action.shape[1]
    
    @property
    def num_continuous_branches(self) -> int:
        """Number of continuous action branches."""
        return self.continuous_action.shape[1]
    
    @property
    def num_branches(self) -> int:
        """Number of total branches."""
        return self.num_discrete_branches + self.num_continuous_branches
    
    @property
    def batch_size(self) -> int:
        """Returns batch_size."""
        return self.discrete_action.shape[0]
    
    @staticmethod
    def to_batch(actions: List["Action"]) -> "Action":
        """Turn action list to single action batch."""
        discrete_action = []
        continuous_action = []
        for a in actions:
            discrete_action.append(a.discrete_action)
            continuous_action.append(a.continuous_action)
            
        action = Action(
            np.concatenate(discrete_action, axis=0),
            np.concatenate(continuous_action, axis=0),
        )
        return action
    
    def to_action_tensor(self, device: Optional[torch.device] = None) -> "ActionTensor":
        """Convert `Action` to `ActionTensor`."""
        return ActionTensor(torch.from_numpy(self.discrete_action).to(device=device), torch.from_numpy(self.continuous_action).to(device=device))
    
    @staticmethod
    def create(discrete_action: Optional[np.ndarray],
               continuous_action: Optional[np.ndarray]) -> "Action":
        if discrete_action is None:
            discrete_action = np.empty(shape=(continuous_action.shape[0], 0))
        if continuous_action is None:
            continuous_action = np.empty(shape=(discrete_action.shape[0], 0))
        
        return Action(discrete_action, continuous_action)
    

class ActionTensor(NamedTuple):
    """
    Standard action data type with tensor. `batch_size` is generally `num_envs` x `n_steps`.

    Args:
        discrete_action (Tensor): shape must be `(batch_size, num_discrete_branches)`
        continuous_action (Tensor): shape must be `(batch_size, num_continuous_branches)`
    """
    
    discrete_action: torch.Tensor
    continuous_action: torch.Tensor
    
    @property
    def num_discrete_branches(self) -> int:
        """Number of discrete action branches."""
        return self.discrete_action.shape[1]
    
    @property
    def num_continuous_branches(self) -> int:
        """Number of continuous action branches."""
        return self.continuous_action.shape[1]
    
    @property
    def num_branches(self) -> int:
        """Number of total branches."""
        return self.num_discrete_branches + self.num_continuous_branches
    
    @property
    def batch_size(self) -> int:
        """Returns batch size."""
        return self.discrete_action.shape[0]
    
    @staticmethod
    def to_batch(actions: List["ActionTensor"]) -> "ActionTensor":
        """Turn action list to single action tensor batch."""
        discrete_action = []
        continuous_action = []
        for a in actions:
            discrete_action.append(a.discrete_action)
            continuous_action.append(a.continuous_action)
            
        action = ActionTensor(
            torch.cat(discrete_action, dim=0),
            torch.cat(continuous_action, dim=0),
        )
        return action
    
    def to_action(self) -> "Action":
        """
        Convert `ActionTensor` to `Action`.
        """
        action = Action(
            self.discrete_action.detach().cpu().numpy(),
            self.continuous_action.detach().cpu().numpy(),
        )
        return action
    
    def slice(self, idx) -> "ActionTensor":
        """
        Slice internal discrete and continuous aaction then merge into single `ActionTensor`. 
        If you want to slice it like `object[idx]`, you should this method instead of directly slicing.
        """
        discrete_action = self.discrete_action[idx]
        continuous_action = self.continuous_action[idx]
        return ActionTensor(discrete_action, continuous_action)
    
    @staticmethod
    def create(discrete_action: Optional[torch.Tensor],
               continuous_action: Optional[torch.Tensor]) -> "ActionTensor":
        """
        Helps to instantiate `ActionTensor`. If you don't use either discrete or continuous action, set the parameter to `None`.
        """
        if discrete_action is None:
            discrete_action = torch.empty(size=(continuous_action.shape[0], 0), device=continuous_action.device)
        if continuous_action is None:
            continuous_action = torch.empty(size=(discrete_action.shape[0], 0), device=discrete_action.device)
        
        action_tensor = ActionTensor(discrete_action, continuous_action)
        return action_tensor

class Experience(NamedTuple):
    """
    Standard experience data type.

    Args:
        obs (ndarray): shape must be `(num_envs, *obs_shape)`
        action (Action): see action shape in `Action` class
        next_obs (ndarray): shape must be `(num_envs, *obs_shape)`
        reward (ndarray): shape must be `(num_envs, 1)`
        terminated (ndarray): shape must be `(num_envs, 1)`
    """
    obs: np.ndarray
    action: Action
    next_obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    
    @property
    def num_envs(self) -> int:
        """Returns number of environments."""
        return self.obs.shape[0]


class ExperienceBatchTensor(NamedTuple):
    """
    Standard experience batch tensor data type. `batch_size` is `num_envs` x `n_steps`.

    Args:
        obs (Tensor): shape must be `(batch_size, *obs_shape)`
        action (ActionTensor): see action shape in `Action` class
        next_obs (Tensor): shape must be `(batch_size, *obs_shape)`
        reward (Tensor): shape must be `(batch_size, 1)`
        terminated (Tensor): shape must be `(batch_size, 1)`
        n_steps (int): number of time steps
    """
    obs: torch.Tensor
    action: ActionTensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    n_steps: int
    