from abc import ABC, abstractmethod
from typing import Optional, Tuple
from aine_drl.policy.policy_distribution import PolicyDistributionParameter
import torch
import torch.nn as nn

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
            obs (torch.Tensor): observation of state whose shape is `(batch_size, *obs_shape)`

        Returns:
            Tuple[PolicyDistributionParameter, torch.Tensor]: policy distribution parameter, state value
        """
        raise NotImplementedError
    
    def train_step(self, 
                   loss: torch.Tensor,
                   grad_clip_max_norm: Optional[float],
                   training_step: int):
        pass
