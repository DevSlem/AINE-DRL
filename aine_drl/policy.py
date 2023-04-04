from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Flag, auto

import torch
import torch.nn as nn

import aine_drl.policy_dist as pd
from aine_drl.util.decay import Decay, NoDecay


class ActionType(Flag):
    DISCRETE = auto()
    CONTINUOUS = auto()
    BOTH = DISCRETE | CONTINUOUS

class Policy(nn.Module, ABC):
    """
    Policy abstract class. It creates policy distribution.
    """
    @property
    @abstractmethod
    def action_type(self) -> ActionType:
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> pd.PolicyDist:
        """
        Create a policy distribution from `pdparam`.

        Args:
            pdparam (PolicyDistParam): policy distribution parameter which is generally the output of neural network

        Returns:
            policy_dist (PolicyDist): policy distribution or action probability distribution
        """
        raise NotImplementedError

class CategoricalPolicy(Policy):
    """
    Categorical policy for the discrete action type.
    
    Args:
        is_logit (bool, optional): wheter discrete parameters are logits or probabilities. Defaults to logit.
    """
    def __init__(
        self,
        in_features: int,
        num_discrete_actions: int | tuple[int, ...],
        bias: bool = True,
        device: torch.device | None = None,
        dtype = None
    ) -> None:
        super().__init__()
        if isinstance(num_discrete_actions, int):
            num_discrete_actions = (num_discrete_actions,)
        self._num_discrete_actions = num_discrete_actions
        self._total_num_discrete_actions = sum(self._num_discrete_actions)
        
        self._layer = nn.Linear(
            in_features,
            self._total_num_discrete_actions,
            bias,
            device,
            dtype
        )
        
    @property
    def action_type(self) -> ActionType:
        return ActionType.DISCRETE
        
    def forward(self, x: torch.Tensor) -> pd.PolicyDist:
        out = self._layer(x)
        logits = out.split(self._num_discrete_actions, dim=-1)
        return pd.CategoricalDist(logits=logits)
    
class GaussianPolicy(Policy):
    """
    Gaussian policy for the continuous action type.
    """
    def __init__(
        self,
        in_features: int,
        num_continous_actions: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype = None
    ) -> None:
        super().__init__()
        
        self._num_continuous_actions = num_continous_actions
        
        self._layer = nn.Linear(
            in_features,
            self._num_continuous_actions * 2,
            bias,
            device,
            dtype
        )
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.CONTINUOUS

    def forward(self, x: torch.Tensor) -> pd.PolicyDist:
        out = self._layer(x)
        # restore batch shape
        mean, std = out.split(self._num_continuous_actions, dim=-1)
        std = std.abs() + 1e-8
        return pd.GaussianDist(mean, std)

class CategoricalGaussianPolicy(Policy):
    """
    Categorical-Gaussian policy for the both discrete and continuous action type.
    
    Args:
        is_logit (bool, optional): wheter discrete parameters are logits or probabilities. Defaults to logit.
    """
    def __init__(
        self,
        in_features: int,
        num_discrete_actions: int | tuple[int, ...],
        num_continous_actions: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype = None
    ) -> None:
        super().__init__()
        
        if isinstance(num_discrete_actions, int):
            num_discrete_actions = (num_discrete_actions,)
        self._num_discrete_actions = num_discrete_actions
        self._total_num_discrete_actions = sum(self._num_discrete_actions)
        self._num_continous_actions = num_continous_actions
        
        self._categorical_layer = nn.Linear(
            in_features,
            self._total_num_discrete_actions,
            bias,
            device,
            dtype
        )
        self._gaussian_layer = nn.Linear(
            in_features,
            self._num_continous_actions * 2,
            bias,
            device,
            dtype
        )
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.BOTH
    
    def forward(self, x: torch.Tensor) -> pd.PolicyDist:
        categorical_out = self._categorical_layer(x)
        gaussian_out = self._gaussian_layer(x)
        # get policy distribution parameter
        logits = categorical_out.split(self._num_discrete_actions, dim=-1)
        mean, std = gaussian_out.split(self._num_continuous_actions, dim=-1)
        std = std.abs() + 1e-8
        return pd.CategoricalGaussianDist(mean=mean, std=std, logits=logits)

class EpsilonGreedyPolicy(Policy):
    """
    Epsilon-greedy policy for value-based method. It only works to the discrete action type.
    
    Args:
        epsilon_decay (float | Decay): epsilon numerical value or decay instance. 0 <= epsilon <= 1
    """
    def __init__(
        self,
        in_features: int,
        num_discrete_actions: int | tuple[int, ...],
        epsilon_decay: float | Decay,
        bias: bool = True,
        device: torch.device | None = None,
        dtype = None
    ) -> None:
        super().__init__()
        
        if isinstance(epsilon_decay, float):
            epsilon_decay = NoDecay(epsilon_decay)
        self._epsilon_decay = epsilon_decay
        self._t = 0
        
        if isinstance(num_discrete_actions, int):
            num_discrete_actions = (num_discrete_actions,)
        self._num_discrete_actions = num_discrete_actions
        self._num_discrete_actions = num_discrete_actions
        self._total_num_discrete_actions = sum(self._num_discrete_actions)
        
        self._layer = nn.Linear(
            in_features,
            self._total_num_discrete_actions,
            bias,
            device,
            dtype
        )
        
        self._action_values = None
        
    @property
    def action_type(self) -> ActionType:
        return ActionType.DISCRETE
        
    def update_t(self, t: float):
        self._t = t
        
    def forward(self, x: torch.Tensor) -> pd.PolicyDist:
        out = self._layer(x)
        self._action_values = out.split(self._num_discrete_actions, dim=-1)
        return pd.EpsilonGreedyDist(self._action_values, self._epsilon_decay(self._t))
    
    def pop_action_values(self) -> tuple[torch.Tensor, ...]:
        action_values = self._action_values
        if action_values is None:
            raise RuntimeError("The action values are not calculated yet.")
        self._action_values = None
        return action_values

class BoltzmannPolicy(Policy):
    """
    TODO: implement
    """
    def __init__(self) -> None:
        raise NotImplementedError

class PolicyActionTypeError(TypeError):
    def __init__(self, valid_action_type: ActionType, invalid_policy: Policy) -> None:
        message = f"The policy action type must be \"{valid_action_type}\", but \"{type(invalid_policy).__name__}\" is \"{invalid_policy.action_type}\"."
        super().__init__(message)
