from abc import ABC, abstractmethod
from typing import List
from aine_drl.trajectory import Trajectory
from aine_drl.drl_algorithm import DRLAlgorithm
from aine_drl.policy import Policy
from aine_drl.util import aine_api
import aine_drl.util as util
from aine_drl.drl_util import Clock, Experience
import numpy as np
import torch

class Agent(ABC):
    """
    Deep reinforcement learning agent.
    """
    def __init__(self, 
                 drl_algorithm: DRLAlgorithm, 
                 policy: Policy, 
                 trajectory: Trajectory,
                 clock: Clock,
                 summary_freq: int = 10) -> None:
        """
        Deep reinforcement learning agent.
        
        Args:
            drl_algorithm (DRLAlgorithm): DRL algorithm
            policy (Policy): policy to sample actions
            trajectory (Trajectory): trajectory to sample training batches
            clock (Clock): time step checker
            summary_freq (int, optional): summary frequency to log data. Defaults to 10.
        """
        self.drl_algorithm = drl_algorithm
        self.policy = policy
        self.trajectory = trajectory
        self.clock = clock
        self.summary_freq = summary_freq
        self.episode_lengths = []
        self.episode_rewards = []
        self.cumulative_rewards = 0
        
    @aine_api
    @abstractmethod
    def train(self, total_time_step: int, start_step: int = 0):
        """Start training.

        Args:
            total_training_step (int): total time step
            start_step (int, optional): training start step. Defaults to 0.
        """
        raise NotImplementedError
        
    @aine_api
    def update(self, experiences: List[Experience]):
        """
        Update the agent. It stores data, trains the DRL algorithm, etc.

        Args:
            experiences (List[Experience]): the number of experiences must be the same as the number of environments.
        """
        # set trajectory
        self.trajectory.add(experiences)
        # update total rewards
        self.cumulative_rewards += experiences[0].reward
        # set clock
        self.clock.tick_time_step()
        if experiences[0].terminated:
            self.episode_lengths.append(self.clock.episode_len)
            self.episode_rewards.append(self.cumulative_rewards)
            self.clock.tick_episode()
            self.cumulative_rewards = 0
        # if can log data
        if self.clock.check_time_step_freq(self.summary_freq):
            time_step = self.clock.time_step
            self._log_data(time_step)
            self.drl_algorithm.log_data(time_step)
            self.policy.log_data(time_step)
        # try training algorithm
        if self._try_train_algorithm():
            time_step = self.clock.time_step
            self.drl_algorithm.update_hyperparams(time_step)
            self.policy.update_hyperparams(time_step)
            
    @aine_api
    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Returns an action from the state.

        Args:
            state (np.ndarray): single state or state batch

        Returns:
            np.ndarray: actions
        """
        pdparam = self.drl_algorithm.get_pdparam(torch.from_numpy(state))
        dist = self.policy.get_policy_distribution(pdparam)
        return dist.sample().cpu().numpy()
    
    def create_experience_list(self,
                               states: np.ndarray,
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
            batch = self.trajectory.sample()
            # train the algorithm
            self.drl_algorithm.train(batch)
            self.clock.tick_training_step()
        return can_train
    
    def _log_data(self, time_step: int):
        if len(self.episode_lengths) > 0:
            average_reward = np.mean(self.episode_rewards)
            print(f"{util.print_title()} training time: {self.clock.real_time:.1f}, time step: {time_step}, average reward: {average_reward:.1f}")
            util.log_data("episode length", np.mean(self.episode_lengths), time_step)
            util.log_data("average reward", average_reward, time_step)
            self.episode_lengths.clear()
            self.episode_rewards.clear()
        else:
            print(f"{util.print_title()} training time: {self.clock.real_time:.1f}, time step: {time_step}, episode has not terminated yet.")
    