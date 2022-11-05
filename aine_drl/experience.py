from typing import NamedTuple, List, Optional
import numpy as np
import torch

class Action(NamedTuple):
    """
    Standard action data type with numpy array. `batch_size` must be `num_envs` x `n_steps`.

    Args:
        discrete_action (ndarray): shape must be `(batch_size, num_discrete_branches)`
        continuous_action (ndarray): shape must be `(batch_size, num_continuous_branches)`
        n_steps (int, optional): number of time steps. Defaults to 1.
    """
    
    discrete_action: np.ndarray
    continuous_action: np.ndarray
    n_steps: int
    
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
        """Batch size = `num_envs` x `n_steps`."""
        return self.discrete_action.shape[0]
    
    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self.batch_size // self.n_steps
    
    @staticmethod
    def to_batch(actions: List["Action"]) -> "Action":
        """Turn action list to single action batch."""
        discrete_action = []
        continuous_action = []
        n_steps = 0
        for a in actions:
            discrete_action.append(a.discrete_action)
            continuous_action.append(a.continuous_action)
            n_steps += a.n_steps
            
        action = Action(
            np.concatenate(discrete_action, axis=0),
            np.concatenate(continuous_action, axis=0),
            n_steps=n_steps
        )
        return action
    
    def to_action_tensor(self) -> "ActionTensor":
        return ActionTensor(torch.from_numpy(self.discrete_action), torch.from_numpy(self.continuous_action), self.n_steps)
    
    @staticmethod
    def create(discrete_action: Optional[torch.Tensor],
               continuous_action: Optional[torch.Tensor],
               n_steps: int = 1) -> "ActionTensor":
        if discrete_action is None:
            discrete_action = np.empty(shape=(continuous_action.shape[0], 0))
        if continuous_action is None:
            continuous_action = np.empty(shape=(discrete_action.shape[0], 0))
        
        return Action(discrete_action, continuous_action, n_steps)
    

class ActionTensor(NamedTuple):
    """
    Standard action data type with tensor. `batch_size` must be `num_envs` x `n_steps`.

    Args:
        discrete_action (Tensor): shape must be `(batch_size, num_discrete_branches)`
        continuous_action (Tensor): shape must be `(batch_size, num_continuous_branches)`
        n_steps (int, optional): number of time steps. Defaults to 1.
    """
    
    discrete_action: torch.Tensor
    continuous_action: torch.Tensor
    n_steps: int
    
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
        """Batch size = `num_envs` x `n_steps`."""
        return self.discrete_action.shape[0]
    
    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self.batch_size // self.n_steps
    
    @staticmethod
    def to_batch(actions: List["ActionTensor"]) -> "ActionTensor":
        """Turn action list to single action tensor batch."""
        discrete_action = []
        continuous_action = []
        n_steps = 0
        for a in actions:
            discrete_action.append(a.discrete_action)
            continuous_action.append(a.continuous_action)
            n_steps += a.n_steps
            
        action = ActionTensor(
            torch.cat(discrete_action, dim=0),
            torch.cat(continuous_action, dim=0),
            n_steps=n_steps
        )
        return action
    
    def to_action(self) -> "Action":
        action = Action(
            self.discrete_action.detach().cpu().numpy(),
            self.continuous_action.detach().cpu().numpy(),
            self.n_steps
        )
        return action
    
    @staticmethod
    def create(discrete_action: Optional[torch.Tensor],
               continuous_action: Optional[torch.Tensor],
               n_steps: int = 1) -> "ActionTensor":
        if discrete_action is None:
            discrete_action = torch.empty(size=(continuous_action.shape[0], 0), device=continuous_action.device)
        if continuous_action is None:
            continuous_action = torch.empty(size=(discrete_action.shape[0], 0), device=discrete_action.device)
        
        return ActionTensor(discrete_action, continuous_action, n_steps)
    

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
    