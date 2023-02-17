from typing import NamedTuple, List, Optional, Callable
from dataclasses import dataclass
import numpy as np
import torch

@dataclass(frozen=True)
class Action:
    """
    Action data type with numpy array.

    Args:
        discrete_action (ndarray): shape must be `(batch_size, num_discrete_branches)`
        continuous_action (ndarray): shape must be `(batch_size, num_continuous_branches)`
    """
    
    discrete_action: np.ndarray
    continuous_action: np.ndarray
    
    def __init__(self,
                 discrete_action: Optional[np.ndarray] = None,
                 continuous_action: Optional[np.ndarray] = None) -> None:
        if discrete_action is None and continuous_action is None:
            raise ValueError("You must input at least one valid argument, but both of them are None.")
        
        if discrete_action is None:
            discrete_action = np.empty((*continuous_action.shape[:-1], 0))
        if continuous_action is None:
            continuous_action = np.empty((*discrete_action.shape[:-1], 0))

        object.__setattr__(self, "discrete_action", discrete_action)
        object.__setattr__(self, "continuous_action", continuous_action)
    
    @property
    def num_discrete_branches(self) -> int:
        """Number of discrete action branches."""
        return self.discrete_action.shape[-1]
    
    @property
    def num_continuous_branches(self) -> int:
        """Number of continuous action branches."""
        return self.continuous_action.shape[-1]
    
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
    
@dataclass(frozen=True)
class ActionTensor:
    """
    Action data type with tensor.

    `*batch_shape` depends on the input of the algorithm you are using. \\
    If it's simple batch, `*batch_shape` = `(batch_size,)`. \\
    If it's sequence batch, `*batch_shape` = `(num_seq, seq_len)`.

    Args:
        discrete_action (Tensor): shape must be `(*batch_shape, num_discrete_branches)`
        continuous_action (Tensor): shape must be `(*batch_shape, num_continuous_branches)`
    """
    
    discrete_action: torch.Tensor
    continuous_action: torch.Tensor
    
    def __init__(self,
                 discrete_action: Optional[torch.Tensor] = None,
                 continuous_action: Optional[torch.Tensor] = None) -> None:
        if discrete_action is None and continuous_action is None:
            raise ValueError("You must input at least one valid argument, but both of them are None.")
        
        if discrete_action is None:
            discrete_action = torch.empty((*continuous_action.shape[:-1], 0), device=continuous_action.device)
        if continuous_action is None:
            continuous_action = torch.empty((*discrete_action.shape[:-1], 0), device=discrete_action.device)

        object.__setattr__(self, "discrete_action", discrete_action)
        object.__setattr__(self, "continuous_action", continuous_action)
        
    @property
    def num_discrete_branches(self) -> int:
        """Number of discrete action branches."""
        return self.discrete_action.shape[-1]
    
    @property
    def num_continuous_branches(self) -> int:
        """Number of continuous action branches."""
        return self.continuous_action.shape[-1]
    
    @property
    def num_branches(self) -> int:
        """Number of total branches."""
        return self.num_discrete_branches + self.num_continuous_branches
    
    @property
    def batch_shape(self) -> torch.Size:
        return self.discrete_action.shape[:-1]
    
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
    
    def transform(self, func: Callable[[torch.Tensor], torch.Tensor]) -> "ActionTensor":
        """
        Transform the action tensor.

        Args:
            func (Callable): function that transforms the action tensor.
        """
        discrete_action = func(self.discrete_action)
        continuous_action = func(self.continuous_action)
        return ActionTensor(discrete_action, continuous_action)
    
    def __getitem__(self, idx) -> "ActionTensor":
        """
        Note that it's recommended to use range slicing instead of indexing.
        """
        discrete_action = self.discrete_action[idx]
        continuous_action = self.continuous_action[idx]
        return ActionTensor(discrete_action, continuous_action)

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
    