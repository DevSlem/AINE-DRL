from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn

class ActorCriticSharedNetwork(nn.Module, ABC):
    """
    Actor critic network.
    """
    
    @abstractmethod
    def forward(self, 
                obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate policy distribution paraemters whose shape is (batch_size, ...), state value whose shape is (batch_size, 1), next_hidden_state which is (h, c) tuple.
        batch_size is generally (num_envs * n-step).
        If action type is discrete, policy distribution is generally logits or softmax distribution.
        If action type is continuous, it's generally mean and standard deviation of gaussian distribution.

        Args:
            obs (torch.Tensor): observation of state whose shape is (batch_size, *obs_shape)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: policy distribution parameter, state value
        """
        raise NotImplementedError
    
    def train_step(self, 
                   loss: torch.Tensor,
                   grad_clip_max_norm: Optional[float],
                   training_step: int):
        pass