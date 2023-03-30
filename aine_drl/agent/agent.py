from abc import ABC, abstractmethod
from enum import Enum

import torch

from aine_drl.exp import Action, Experience, Observation
from aine_drl.net import Network


class BehaviorType(Enum):
    TRAIN = 0,
    INFERENCE = 1

class Agent(ABC):
    """
    Deep reinforcement learning agent.
    
    Args:
        network (Network): deep neural network
        policy (Policy): policy
        num_envs (int): number of environments
    """
    def __init__(
        self,
        num_envs: int,
        network: Network,
        behavior_type: BehaviorType = BehaviorType.TRAIN
    ) -> None:
        assert num_envs >= 1, "The number of environments must be greater than or euqal to 1."
        
        self._device = network.device
        self._model = network.model()
        self._num_envs = num_envs
        self._behavior_type = behavior_type
        
        self._training_steps = 0
        
    def select_action(self, obs: Observation) -> Action:
        """
        Select actions from the `obs`.

        Args:
            obs (Tensor): observation `(num_envs, *obs_shape)`

        Returns:
            action (Tensor): `*batch_shape` = `(num_envs,)`
        """
        match self.behavior_type:
            case BehaviorType.TRAIN:
                return self._select_action_train(obs).transform(torch.detach)
            case BehaviorType.INFERENCE:
                return self._select_action_inference(obs).transform(torch.detach)
            
    def update(self, exp: Experience):
        """
        Update and train the agent.

        Args:
            exp (Experience): one-step experience tuple
        """ 
        match self.behavior_type:
            case BehaviorType.TRAIN:
                self._update_train(exp)
            case BehaviorType.INFERENCE:
                self._update_inference(exp)
                
    @abstractmethod
    def _update_train(self, exp: Experience):
        raise NotImplementedError
    
    @abstractmethod
    def _update_inference(self, exp: Experience):
        raise NotImplementedError
              
    @abstractmethod
    def _select_action_train(self, obs: Observation) -> Action:
        raise NotImplementedError
    
    @abstractmethod
    def _select_action_inference(self, obs: Observation) -> Action:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def training_steps(self) -> int:
        return self._training_steps
    
    def _tick_training_steps(self):
        self._training_steps += 1
    
    @property
    def behavior_type(self) -> BehaviorType:
        """Returns behavior type. Defaults to train."""
        return self._behavior_type
    
    @behavior_type.setter
    def behavior_type(self, value: BehaviorType):
        """Set behavior type."""
        self._behavior_type = value
        match self._behavior_type:
            case BehaviorType.TRAIN:
                self._model.train()
            case BehaviorType.INFERENCE:
                self._model.eval()
    
    @property
    def log_keys(self) -> tuple[str, ...]:
        """Returns log data keys."""
        return tuple()
        
    @property
    def log_data(self) -> dict[str, tuple]:
        """
        Returns log data and reset it.

        Returns:
            dict[str, tuple]: key: (value, time)
        """
        return dict()
            
    @property
    def state_dict(self) -> dict:
        """Returns the state dict of the agent."""
        return dict(
            training_steps=self._training_steps,
            model=self._model.state_dict()
        )
    
    def load_state_dict(self, state_dict: dict):
        """Load the state dict."""
        self._training_steps = state_dict["training_steps"]
        self._model.load_state_dict(state_dict["model"])

class BehaviorScope:
    def __init__(self, agent: Agent, behavior_type: BehaviorType) -> None:
        self._agent = agent
        self._old_behavior_type = agent.behavior_type
        self._new_behavior_type = behavior_type
        
    def __enter__(self):
        self._agent.behavior_type = self._new_behavior_type
        
    def __exit__(self, exc_type, exc_value, traceback):
        self._agent._behavior_type = self._old_behavior_type