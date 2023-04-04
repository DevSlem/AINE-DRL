from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class Observation:
    """
    Observation data type with tensor tuple. It can have multiple observation spaces.

    `*batch_shape` depends on the input of the algorithm you are using.

    * simple batch: `*batch_shape` = `(batch_size,)`
    * sequence batch: `*batch_shape` = `(seq_batch_size, seq_len)`
    
    You can subscript the `Observation` instance to get a batch of `Observation`.
    Note that it works to all observation `Tensor` instances.
    
    Args:
        items (tuple[Tensor, ...]): tuple of observation tensors
    """
    items: tuple[Tensor, ...]
    
    @property
    def num_items(self) -> int:
        return len(self.items)
    
    def transform(self, func: Callable[[Tensor], Tensor]) -> "Observation":
        """
        Transform each observation tensor.

        Args:
            func (Callable[[Tensor], Tensor]): function to transform observation tensor

        Returns:
            transformed_obs (Observation): transformed observation
        """
        return Observation(tuple(func(obs_tensor) for obs_tensor in self.items))
    
    def clone(self) -> "Observation":
        return self.transform(lambda o: o.clone())
    
    def __getitem__(self, idx) -> "Observation":
        """
        Note that it's recommended to use range slicing instead of indexing.
        """
        return Observation(tuple(obs_tensor[idx] for obs_tensor in self.items))
    
    def __setitem__(self, idx, value: "Observation"):
        """
        Note that it's recommended to use range slicing instead of indexing.
        """
        for obs_tensor, value_obs_tensor in zip(self.items, value.items):
            obs_tensor[idx] = value_obs_tensor
    
    @staticmethod
    def from_tensor(*obs_tensors: Tensor) -> "Observation":
        """
        Create Observation from tensors.
        """
        return Observation(obs_tensors)
    
    @staticmethod
    def from_iter(iter: Iterable["Observation"]) -> "Observation":
        """
        Create Observation batch from iterable of Observation.

        Args:
            iter (Iterable[Observation]): iterable of Observation
        """
        unwrapped = tuple(obs.items for obs in iter)
        return Observation(
            tuple(torch.cat(obs_tensors, dim=0) for obs_tensors in zip(*unwrapped))
        )

@dataclass(frozen=True)
class Action:
    """
    Action data type with tensor.

    `*batch_shape` depends on the input of the algorithm you are using.
    
    * simple batch: `*batch_shape` = `(batch_size,)`
    * sequence batch: `*batch_shape` = `(seq_batch_size, seq_len)`

    Args:
        discrete_action (Tensor): `(*batch_shape, num_discrete_branches)`
        continuous_action (Tensor): `(*batch_shape, num_continuous_branches)`
    """
    discrete_action: Tensor
    continuous_action: Tensor
    
    def __init__(
        self,
        discrete_action: Tensor | None = None,
        continuous_action: Tensor | None = None
    ) -> None:
        if discrete_action is None and continuous_action is None:
            raise ValueError("You must input at least one valid argument, but both of them are None.")
        
        if discrete_action is None:
            discrete_action = torch.empty((*continuous_action.shape[:-1], 0), device=continuous_action.device) # type: ignore
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
        Transform both discrete and continuous actions.

        Args:
            func (Callable): function that transforms each action.
        """
        discrete_action = func(self.discrete_action)
        continuous_action = func(self.continuous_action)
        return Action(discrete_action, continuous_action)
    
    def clone(self) -> "Action":
        return self.transform(lambda a: a.clone())
    
    def __getitem__(self, idx) -> "Action":
        """
        Note that it's recommended to use range slicing instead of indexing.
        """
        discrete_action = self.discrete_action[idx]
        continuous_action = self.continuous_action[idx]
        return Action(discrete_action, continuous_action)
    
    @staticmethod
    def from_iter(actions: Iterable["Action"]) -> "Action":
        """Create Action batch from iterable of Action."""
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
        obs (Observation): see observation shape in `Observation` class
        action (Action): see action shape in `Action` class
        next_obs (Observation): see observation shape in `Observation` class
        reward (Tensor): `(num_envs, 1)`
        terminated (Tensor): `(num_envs, 1)`
    """
    obs: Observation
    action: Action
    next_obs: Observation
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
            obs=self.obs.transform(func),
            action=self.action.transform(func),
            next_obs=self.next_obs.transform(func),
            reward=func(self.reward),
            terminated=func(self.terminated)
        )
        