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
        pass
    
    @aine_api
    @property
    @abstractmethod
    def can_train(self) -> bool:
        """
        Returns wheter you can train or not.
        """
        pass
    
    @aine_api
    @abstractmethod
    def reset(self):
        """
        Reset the trajectory.
        """
        pass
    
    @aine_api
    @abstractmethod
    def add(self, experiences: Union[Experience, List[Experience]]):
        """Add one or more experiences.

        Args:
            experiences (Experience | List[Experience]): 
            both single or list are okay, but be sure that count equals to environment count
        """
        pass
        
    @aine_api
    @abstractmethod
    def sample(self) -> ExperienceBatch:
        """Sample from the trajectory.

        Returns:
            ExperienceBatch: sampled experience batch
        """
        pass
