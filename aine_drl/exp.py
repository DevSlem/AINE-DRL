from typing import Callable, Iterable
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass(frozen=True)
class Action:
    """
    Action data type with tensor.

    `*batch_shape` depends on the input of the algorithm you are using. \\
    If it's simple batch, `*batch_shape` = `(batch_size,)`. \\
    If it's sequence batch, `*batch_shape` = `(seq_batch_size, seq_len)`.

    Args:
        discrete_action (Tensor): `(*batch_shape, num_discrete_branches)`
        continuous_action (Tensor): `(*batch_shape, num_continuous_branches)`
    """
    
    discrete_action: Tensor
    continuous_action: Tensor
    
    def __init__(self,
                 discrete_action: Tensor | None = None,
                 continuous_action: Tensor | None = None) -> None:
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
    
    def transform(self, func: Callable[[Tensor], Tensor]) -> "Action":
        """
        Transform the action.

        Args:
            func (Callable): function that transforms the action.
        """
        discrete_action = func(self.discrete_action)
        continuous_action = func(self.continuous_action)
        return Action(discrete_action, continuous_action)
    
    def __getitem__(self, idx) -> "Action":
        """
        Note that it's recommended to use range slicing instead of indexing.
        """
        discrete_action = self.discrete_action[idx]
        continuous_action = self.continuous_action[idx]
        return Action(discrete_action, continuous_action)
    
    @staticmethod
    def from_iter(actions: Iterable["Action"]) -> "Action":
        """Make action batch from the action list."""
        discrete_action = []
        continuous_action = []
        for a in actions:
            discrete_action.append(a.discrete_action)
            continuous_action.append(a.continuous_action)
                    
        return Action(
            torch.cat(discrete_action, dim=0),
            torch.cat(continuous_action, dim=0),
        )

@dataclass(frozen=True)
class Experience:
    """
    Experience data type.

    Args:
        obs (Tensor): `(num_envs, *obs_shape)`
        action (Action): see action shape in `Action` class
        next_obs (Tensor): `(num_envs, *obs_shape)`
        reward (Tensor): `(num_envs, 1)`
        terminated (Tensor): `(num_envs, 1)`
    """
    obs: Tensor
    action: Action
    next_obs: Tensor
    reward: Tensor
    terminated: Tensor
    
    def transform(self, func: Callable[[Tensor], Tensor]) -> "Experience":
        """
        Transform the experience.

        Args:
            func (Callable[[TSource], TResult]): transform function

        Returns:
            Experience[TResult]: transformed experience
        """
        return Experience(
            obs=func(self.obs),
            action=self.action.transform(func),
            next_obs=func(self.next_obs),
            reward=func(self.reward),
            terminated=func(self.terminated)
        )
        