from abc import ABC, abstractmethod
from typing import List
from aine_drl import aine_api
from aine_drl.drl_util import Experience, ExperienceBatch

class Trajectory(ABC):
    """
    Trajectory abstract class to store and sample experiences.
    """
    
    @aine_api
    @property
    @abstractmethod
    def count(self) -> int:
        """
        Returns stored experiences count.
        """
        raise NotImplementedError
    
    @aine_api
    @property
    @abstractmethod
    def can_train(self) -> bool:
        """
        Returns wheter you can train or not.
        """
        raise NotImplementedError
    
    @aine_api
    @abstractmethod
    def reset(self):
        """
        Reset the trajectory.
        """
        raise NotImplementedError
    
    @aine_api
    @abstractmethod
    def add(self, experiences: List[Experience]):
        """
        Add experiences from the one-step transition. The number of experiences must be the same as the number of environments.

        Args:
            experiences (List[Experience]): experiences of environments
        """
        raise NotImplementedError
        
    @aine_api
    @abstractmethod
    def sample(self) -> ExperienceBatch:
        """
        Sample from the trajectory. You should call this function only while can train.

        Returns:
            ExperienceBatch: sampled experience batch
        """
        raise NotImplementedError
