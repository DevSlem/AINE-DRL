from abc import ABC, abstractmethod
from typing import Any, Union
from aine_drl.experience import Action
import gym.spaces as gym_space
from gym import Env
from gym.vector import VectorEnv
from gym.spaces import Space

class GymActionCommunicator(ABC):
    """Action communicator between AINE-DRL and Gym."""
    @abstractmethod
    def to_gym_action(self, action: Action) -> Any:
        raise NotImplementedError
    
    @staticmethod
    def make(gym_env: Union[Env, VectorEnv]) -> "GymActionCommunicator":
        """Make automatically gym action communicator."""
        action_space_type = type(gym_env.action_space)
        if action_space_type is gym_space.Discrete or action_space_type is gym_space.MultiDiscrete:
            return GymDiscreteActionCommunicator(gym_env.action_space)
        elif action_space_type is gym_space.Box:
            return GymContinuousActionCommunicator(gym_env.action_space)
        else:
            raise ValueError("Doesn't implement any gym action communicator for this action space yet. You need to set it manually.")
    
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
