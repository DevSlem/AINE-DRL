from abc import ABC, abstractmethod
from typing import Any
from aine_drl.experience import Action
from gym.spaces import Space

class GymActionCommunicator(ABC):
    """Action communicator between AINE-DRL and Gym."""
    @abstractmethod
    def to_gym_action(self, action: Action) -> Any:
        raise NotImplementedError
    
class GymDiscreteActionCommunicator(GymActionCommunicator):
    def __init__(self, action_space: Space) -> None:
        self.action_shape = action_space.shape
    
    def to_gym_action(self, action: Action) -> Any:
        return action.discrete_action.reshape(self.action_shape)

class GymContinuousActionCommunicator(GymActionCommunicator):
    def __init__(self, action_space: Space) -> None:
        super().__init__()
        self.action_shape = action_space.shape
    
    def to_gym_action(self, action: Action) -> Any:
        return action.continuous_action.reshape(self.action_shape)
