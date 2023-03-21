from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum

import torch

from aine_drl.drl_util import Clock, IClockNeed, ILogable
from aine_drl.exp import Action, Experience
from aine_drl.net import Network
from aine_drl.policy.policy import Policy


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
        network: Network,
        policy: Policy,
        num_envs: int,
        behavior_type: BehaviorType = BehaviorType.TRAIN
    ) -> None:
        assert num_envs >= 1, "The number of environments must be greater than or euqal to 1."
        
        self._clock = Clock(num_envs)
        if isinstance(network, IClockNeed):
            network.set_clock(self._clock)
        if isinstance(policy, IClockNeed):
            policy.set_clock(self._clock)
            
        self.__logable_policy = policy if isinstance(policy, ILogable) else None
        
        self._policy = policy
        self._network = network
        self._num_envs = num_envs
        self._behavior_type = behavior_type
        
        self._using_behavior_type_scope = False
        
    def select_action(self, obs: torch.Tensor) -> Action:
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
                self._update_train_info(exp)
                self._update_train(exp)
            case BehaviorType.INFERENCE:
                self._update_inference(exp)
                
    def _update_train_info(self, exp: Experience):
        self.clock.tick_gloabl_time_step()
    
    @abstractmethod
    def _update_train(self, exp: Experience):
        raise NotImplementedError
    
    @abstractmethod
    def _update_inference(self, exp: Experience):
        raise NotImplementedError
              
    @abstractmethod
    def _select_action_train(self, obs: torch.Tensor) -> Action:
        raise NotImplementedError
    
    @abstractmethod
    def _select_action_inference(self, obs: torch.Tensor) -> Action:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
    
    @property
    def device(self) -> torch.device:
        return self._network.device
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def clock(self) -> Clock:
        return self._clock
    
    @property
    def behavior_type(self) -> BehaviorType:
        """Returns behavior type. Defaults to train."""
        return self._behavior_type
    
    @behavior_type.setter
    def behavior_type(self, value: BehaviorType):
        """Set behavior type."""
        self._behavior_type = value
        
    @contextmanager
    def behavior_type_scope(self, behavior_type: BehaviorType):
        """
        Context manager for behavior type.
        
        Example::
        
            with agent.behavior_type_scope(BehaviorType.INFERENCE):
                # do something
        """
        self._using_behavior_type_scope = True
        old_behavior_type = self.behavior_type
        self.behavior_type = behavior_type
        yield
        self._using_behavior_type_scope = False
        self.behavior_type = old_behavior_type
    
    @property
    def log_keys(self) -> tuple[str, ...]:
        """Returns log data keys."""
        lk = tuple()
        if self.__logable_policy is not None:
            lk += self.__logable_policy.log_keys
        return lk
        
    @property
    def log_data(self) -> dict[str, tuple]:
        """
        Returns log data and reset it.

        Returns:
            dict[str, tuple]: key: (value, time)
        """
        ld = {}
        if self.__logable_policy is not None:
            ld.update(self.__logable_policy.log_data)
        return ld
            
    @property
    def state_dict(self) -> dict:
        """Returns the state dict of the agent."""
        return dict(
            clock=self.clock.state_dict
        )
    
    def load_state_dict(self, state_dict: dict):
        """Load the state dict."""
        self.clock.load_state_dict(state_dict["clock"])
