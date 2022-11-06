from abc import ABC, abstractmethod
from typing import Tuple
from aine_drl.experience import Action, ActionTensor, Experience
import aine_drl.util as util
from aine_drl.drl_util import Clock
import numpy as np
import torch
from enum import Enum

class BehaviorType(Enum):
    TRAIN = 0,
    INFERENCE = 1

class Agent(ABC):
    """
    Deep reinforcement learning agent.
    """
    def __init__(self, num_envs: int) -> None:
        """
        Deep reinforcement learning agent.
        
        Args:
            num_envs (int): number of environments
        """
        self.num_envs = num_envs
        self.clock = Clock(num_envs)
        self._behavior_type = BehaviorType.TRAIN

        self.traced_env = 0
        self.cumulative_average_reward = util.IncrementalAverage()
        self.cumulative_reward = 0.0
        self.episode_average_len = util.IncrementalAverage()
        
    def select_action(self, obs: np.ndarray) -> Action:
        """
        Select action from the observation. 
        `batch_size` must be `num_envs` x `n_steps`. `n_steps` is generally 1. 
        It depends on `Agent.behavior_type` enum value.

        Args:
            obs (ndarray): observation which is the input of neural network. shape must be `(batch_size, *obs_shape)`

        Returns:
            Action: selected action
        """
        if self.behavior_type == BehaviorType.TRAIN:
            return self.select_action_train(torch.from_numpy(obs)).to_action()
        elif self.behavior_type == BehaviorType.INFERENCE:
            with torch.no_grad():
                return self.select_action_inference(torch.from_numpy(obs)).to_action()
        else:
            raise ValueError(f"Agent.behavior_type you currently use is invalid value. Your value is: {self.behavior_type}")
        
    def update(self, experience: Experience):
        """
        Update the agent. It stores data, trains the agent, etc.

        Args:
            experience (Experience): experience
        """
        assert experience.num_envs == self.num_envs
        
        self._update_info(experience)
            
    def _update_info(self, experience: Experience):
        self.clock.tick_gloabl_time_step()
        self.cumulative_reward += experience.reward[self.traced_env].item()
        # if the traced environment is terminated
        if experience.terminated[self.traced_env] > 0.5:
            self.cumulative_average_reward.update(self.cumulative_reward)
            self.cumulative_reward = 0.0
            self.episode_average_len.update(self.clock.episode_len)
            self.clock.tick_episode() 
        
    @abstractmethod
    def select_action_train(self, obs: torch.Tensor) -> ActionTensor:
        """
        Select action when training.

        Args:
            obs (Tensor): observation tensor whose shape is `(batch_size, *obs_shape)`

        Returns:
            ActionTensor: action tensor
        """
        raise NotImplementedError
    
    @abstractmethod
    def select_action_inference(self, obs: torch.Tensor) -> ActionTensor:
        """
        Select action when inference. It's automatically called with `torch.no_grad()`.

        Args:
            obs (Tensor): observation tensor

        Returns:
            ActionTensor action tensor
        """
        raise NotImplementedError
    
    @property
    def log_keys(self) -> Tuple[str, ...]:
        """Returns log data keys."""
        return ("Environment/Cumulative Reward", "Environment/Episode Length")
        
    @property
    def log_data(self) -> dict:
        """Returns log data and reset it."""
        ld = {}
        if self.cumulative_average_reward.count > 0:
            ld["Environment/Cumulative Reward"] = self.cumulative_average_reward.average
            ld["Environment/Episode Length"] = self.episode_average_len.average
            self.cumulative_average_reward.reset()
            self.episode_average_len.reset()
        return ld
            
    @property
    def behavior_type(self) -> BehaviorType:
        """Returns behavior type. Defaults to train."""
        return self._behavior_type
    
    @behavior_type.setter
    def behavior_type(self, value: BehaviorType):
        """Set behavior type."""
        self._behavior_type = value
            
    @property
    def state_dict(self) -> dict:
        """Returns the state dict of the agent."""
        return self.clock.state_dict
    
    def load_state_dict(self, state_dict: dict):
        """Load the state dict."""
        self.clock.load_state_dict(state_dict)
