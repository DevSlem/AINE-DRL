from abc import ABC, abstractmethod
from typing import Union, List
from aine_drl import aine_api
from aine_drl.experience import Experience, ExperienceBatch

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
    def add(self, experiences: Union[Experience, List[Experience]]):
        """
        Add one or more experiences.
        
        Args:
            experiences (Experience | List[Experience]): 
            both single or list are okay, but be sure that count equals to environment count
        """
        raise NotImplementedError
        
    @aine_api
    @abstractmethod
    def sample(self) -> ExperienceBatch:
        """
        Sample from the trajectory. You should call this function only if can train.

        Returns:
            ExperienceBatch: sampled experience batch
        """
        raise NotImplementedError
