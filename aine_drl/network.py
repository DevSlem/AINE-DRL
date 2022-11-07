from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Any
from aine_drl.policy.policy_distribution import PolicyDistributionParameter
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteActionLayer(nn.Module):
    """
    Linear layer for the discrete action type.

    Args:
        in_features (int): number of input features
        num_discrete_actions (int | Tuple[int, ...]): each element indicates number of discrete actions of each branch
        is_logits (bool): whether logits or probabilities. Defaults to logits.
    """

    def __init__(self, in_features: int, 
                 num_discrete_actions: Union[int, Tuple[int, ...]], 
                 is_logits: bool = True,
                 bias: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[Any] = None) -> None:
        """
        Linear layer for the discrete action type.

        Args:
            in_features (int): number of input features
            num_discrete_actions (int | Tuple[int, ...]): each element indicates number of discrete actions of each branch
            is_logits (bool): whether logits or probabilities. Defaults to logits.
        """
        super().__init__()
        
        self.is_logits = is_logits
        
        if type(num_discrete_actions) is int:
            num_discrete_actions = (num_discrete_actions,)
        self.num_discrete_actions = num_discrete_actions
        
        self.total_num_discrete_actions = 0
        for num_action in num_discrete_actions:
            self.total_num_discrete_actions += num_action
        
        self.layer = nn.Linear(
            in_features,
            self.total_num_discrete_actions,
            bias,
            device,
            dtype
        )
    
    def forward(self, x: torch.Tensor) -> PolicyDistributionParameter:
        out = self.layer(x)
        discrete_pdparams = list(torch.split(out, self.num_discrete_actions, dim=1))
        
        if not self.is_logits:
            for i in range(len(discrete_pdparams)):
                discrete_pdparams[i] = F.softmax(discrete_pdparams[i], dim=1)
        
        return PolicyDistributionParameter.create(discrete_pdparams, None)
    
class GaussianContinuousActionLayer(nn.Module):
    """
    Linear layer for the continuous action type.

    Args:
        in_features (int): number of input features
        num_continuous_actions (int): number of continuous actions which equals to `num_continuous_branches`
    """
    
    def __init__(self, in_features: int, 
                 num_continuous_actions: int, 
                 bias: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[Any] = None) -> None:
        """
        Linear layer for the continuous action type.

        Args:
            in_features (int): number of input features
            num_continuous_actions (int): number of continuous actions which equals to `num_continuous_branches`
        """
        super().__init__()
        
        self.num_continuous_actions = num_continuous_actions
        
        self.layer = nn.Linear(
            in_features,
            self.num_continuous_actions * 2,
            bias,
            device,
            dtype
        )
        
    def forward(self, x: torch.Tensor) -> PolicyDistributionParameter:
        out = self.layer(x)
        out = torch.reshape(out, (-1, self.num_continuous_actions, 2))
        torch.abs_(out[..., 1])
        out = torch.reshape(out, (-1, self.num_continuous_actions * 2))
        continuous_pdparams = list(torch.split(out, 2, dim=1))
        return PolicyDistributionParameter.create(None, continuous_pdparams)
    
class PolicyGradientNetwork(nn.Module, ABC):
    """
    Policy gradient network.
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> PolicyDistributionParameter:
        """
        Calculate policy distribution paraemters whose shape is `(batch_size, ...)`. \\
        `batch_size` is `num_envs` x `n-step`. \\
        When the action type is discrete, policy distribution is generally logits or soft-max distribution. \\
        When the action type is continuous, it's generally mean and standard deviation of gaussian distribution.

        Args:
            obs (Tensor): observation of state whose shape is `(batch_size, *obs_shape)`

        Returns:
            PolicyDistributionParameter: policy distribution parameter
        """
        raise NotImplementedError
    
    @abstractmethod
    def train_step(self, 
                   loss: torch.Tensor,
                   grad_clip_max_norm: Optional[float],
                   training_step: int):
        raise NotImplementedError

class ActorCriticSharedNetwork(nn.Module, ABC):
    """
    Actor critic network.
    """
    
    @abstractmethod
    def forward(self, 
                obs: torch.Tensor) -> Tuple[PolicyDistributionParameter, torch.Tensor]:
        """
        Calculate policy distribution paraemters whose shape is `(batch_size, ...)`, 
        state value whose shape is `(batch_size, 1)`. \\
        `batch_size` is `num_envs` x `n-step`. \\
        When the action type is discrete, policy distribution is generally logits or soft-max distribution. \\
        When the action type is continuous, it's generally mean and standard deviation of gaussian distribution.

        Args:
            obs (Tensor): observation of state whose shape is `(batch_size, *obs_shape)`

        Returns:
            Tuple[PolicyDistributionParameter, Tensor]: policy distribution parameter, state value
        """
        raise NotImplementedError
    
    @abstractmethod
    def train_step(self, 
                   loss: torch.Tensor,
                   grad_clip_max_norm: Optional[float],
                   training_step: int):
        raise NotImplementedError
