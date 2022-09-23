from abc import ABC, abstractmethod
import torch
from torch.distributions import Distribution

class Policy(ABC):
    """
    Policy abstract class. It returns policy distribution.
    """
    
    @abstractmethod
    def get_policy_distribution(self, pdparam: torch.Tensor) -> Distribution:
        """
        Returns policy distribution that is action probability distribution.

        Args:
            pdparam (torch.Tensor): policy distribution parameter that is noramlly the output of neural network

        Returns:
            Distribution: policy distribution
        """
        raise NotImplementedError
    
    def update_hyperparams(self, time_step: int):
        """
        Update hyperparameters if they exists.

        Args:
            time_step (int): current time step during training
        """
        pass
    
    def log_data(self, time_step: int):
        pass
