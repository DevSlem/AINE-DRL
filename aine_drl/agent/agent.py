from abc import ABC, abstractmethod
from typing import List
from aine_drl.drl_util.net_spec import NetSpec
from aine_drl.trajectory import Trajectory
from aine_drl.policy import Policy
from aine_drl.util import logger
from aine_drl.drl_util import Clock, Experience
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
    def __init__(self,
                 net_spec: NetSpec,
                 policy: Policy, 
                 trajectory: Trajectory,
                 clock: Clock,
                 summary_freq: int = 1000) -> None:
        """
        Deep reinforcement learning agent.
        
        Args:
            net_spec (NetSpec): neural network spec
            policy (Policy): policy to sample actions
            trajectory (Trajectory): trajectory to sample training batches
            clock (Clock): time step checker
            summary_freq (int, optional): summary frequency to log data. Defaults to 10.
        """
        self.__net_spec = net_spec
        self.policy = policy
        self.trajectory = trajectory
        self.clock = clock
        self.summary_freq = summary_freq
        self.episode_lengths = []
        self.cumulative_rewards = []
        self.cumulative_reward = 0
        self._behavior_type = BehaviorType.TRAIN
        
    def update(self, experiences: List[Experience]):
        """
        Update the agent. It stores data, trains the agent, etc.

        Args:
            experiences (List[Experience]): the number of experiences must be the same as the number of environments.
        """
        # update trajectory
        self.trajectory.add(experiences)
        # trace cumulative rewards
        self.cumulative_reward += experiences[0].reward
        # set clock
        self.clock.tick_time_step()
        # try training
        if self._try_train_algorithm():
            time_step = self.clock.time_step
            self.update_hyperparams(time_step)
        # trace first experience
        if experiences[0].terminated:
            self.episode_lengths.append(self.clock.episode_len)
            self.cumulative_rewards.append(self.cumulative_reward)
            self.clock.tick_episode()
            self.cumulative_reward = 0
        # if can log data
        if self.clock.check_time_step_freq(self.summary_freq):
            time_step = self.clock.time_step
            self.log_data(time_step)
            self.policy.log_data(time_step)
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns an action from the state.

        Args:
            state (np.ndarray): single state or states batch

        Returns:
            np.ndarray: single action or actions batch
        """
        if self.behavior_type == BehaviorType.TRAIN:
            return self.select_action_tensor(torch.from_numpy(state)).cpu().numpy()
        elif self.behavior_type == BehaviorType.INFERENCE:
            with torch.no_grad():
                return self.select_action_inference(torch.from_numpy(state)).cpu().numpy()
    
    @abstractmethod
    def train(self):
        """
        Train the agent. You need to implement a reinforcement learning algorithm.
        """
        raise NotImplementedError
        
    @abstractmethod
    def select_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select actions when training.

        Args:
            state (torch.Tensor): state observation tensor

        Returns:
            torch.Tensor: action tensor
        """
        raise NotImplementedError
    
    @abstractmethod
    def select_action_inference(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select actions when inference. It's called with `torch.no_grad()`.

        Args:
            state (torch.Tensor): state observation tensor

        Returns:
            torch.Tensor: action tensor
        """
        raise NotImplementedError
    
    def update_hyperparams(self, time_step: int):
        """
        Update hyperparameters if they exists.

        Args:
            time_step (int): current time step during training
        """
        self.policy.update_hyperparams(time_step)
    
    def log_data(self, time_step: int):
        """Log data."""
        if len(self.episode_lengths) > 0:
            avg_cumul_reward = np.mean(self.cumulative_rewards)
            logger.print(f"training time: {self.clock.real_time:.1f}, time step: {time_step}, cumulative reward: {avg_cumul_reward:.1f}")
            logger.log("Environment/Cumulative Reward", avg_cumul_reward, time_step)
            logger.log("Environment/Cumulative Reward per episodes", avg_cumul_reward, self.clock.episode)
            logger.log("Environment/Episode Length", np.mean(self.episode_lengths), time_step)
            logger.log("Environment/Episode Length per episodes", np.mean(self.episode_lengths), self.clock.episode)
            self.episode_lengths.clear()
            self.cumulative_rewards.clear()
        else:
            logger.print(f"training time: {self.clock.real_time:.1f}, time step: {time_step}, episode has not terminated yet.")
            
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
        return {"clock": self.clock.state_dict, "net_spec": self.__net_spec.state_dict}
    
    def load_state_dict(self, state_dict: dict):
        """Load the state dict."""
        self.clock.load_state_dict(state_dict["clock"])
        self.__net_spec.load_state_dict(state_dict["net_spec"])
    
    @staticmethod
    def create_experience_list(states: np.ndarray,
                               actions: np.ndarray,
                               next_states: np.ndarray,
                               rewards: np.ndarray,
                               terminateds: np.ndarray) -> List[Experience]:
        """
        Changes the numpy batch to the list of Experience instances.

        Args:
            states (np.ndarray): state batch
            actions (np.ndarray): action batch
            next_states (np.ndarray): next_state batch
            rewards (np.ndarray): reward batch
            terminateds (np.ndarray): terminated batch

        Returns:
            List[Experience]: list of Experience instances
        """
        exp_list = []
        for s, a, ns, r, t in zip(states, actions, next_states, rewards, terminateds):
            exp_list.append(Experience(s, a, ns, r, t))
        return exp_list
    
    def _try_train_algorithm(self) -> bool:
        can_train = self.trajectory.can_train
        while self.trajectory.can_train:
            self.train()
        return can_train
