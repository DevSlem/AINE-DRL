from abc import ABC, abstractmethod
from aine_drl.trajectory import ExperienceBatch
from aine_drl.util import aine_api
import torch

class DRLAlgorithm(ABC):
    """
    Deep Reinforcement Learning Algorithm abstract class.
    """
    
    @aine_api
    @abstractmethod
    def train(self, batch: ExperienceBatch):
        """
        Train the algorithm.

        Args:
            batch (ExperienceBatch): training data
        """
        raise NotImplementedError
    
    @aine_api
    @abstractmethod
    def get_pdparam(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns policy distribution parameter which is typically the output of the neural network.

        Args:
            state (torch.Tensor): to get action data for the state

        Returns:
            torch.Tensor: policy distribution parameter
        """
        raise NotImplementedError
    
    @aine_api
    def update_hyperparams(self, time_step: int):
        """
        Update hyperparameters if they exists.

        Args:
            time_step (int): current time step during training
        """
        pass
    
    @aine_api
    def log_data(self, time_step: int):
        pass
