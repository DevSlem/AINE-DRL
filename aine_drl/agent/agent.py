import enum
from typing import List, Tuple
from aine_drl.trajectory import Trajectory
from aine_drl.drl_algorithm import DRLAlgorithm
from aine_drl.policy import Policy
from aine_drl.util import aine_api
import aine_drl.util as util
from aine_drl.drl_util import Clock, Experience
import numpy as np
import torch

class Agent:
    def __init__(self, 
                 drl_algorithm: DRLAlgorithm, 
                 policy: Policy, 
                 trajectory: Trajectory,
                 clock: Clock,
                 summary_freq: int = 10) -> None:
        """
        Args:
            drl_algorithm (DRLAlgorithm): DRL algorithm
            policy (Policy): policy
            trajectory (Trajectory): tajectory
            summary_freq (int, optional): summary frequency. Defaults to 1.
        """
        self.drl_algorithm = drl_algorithm
        self.policy = policy
        self.trajectory = trajectory
        self.clock = clock
        self.summary_freq = summary_freq
        self.episode_lengths = []
        self.episode_rewards = []
        self._reset_total_rewards()
        
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
        for i, exp in enumerate(experiences):
            self.total_rewards[i] += exp.reward
        # set clock
        self.clock.tick_time_step()
        terminated_exp = self._terminated_exp(experiences)
        # if any environment is terminated
        if terminated_exp >= 0:
            if self.clock.episode_len > 500:
                print(f"episode length: {self.clock.episode_len}")
            self.episode_lengths.append(self.clock.episode_len)
            self.episode_rewards.append(self.total_rewards[terminated_exp])
            self.clock.tick_episode()
            self._reset_total_rewards()
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
            util.log_data("episode length", np.mean(self.episode_lengths), time_step)
            util.log_data("average reward", np.mean(self.episode_rewards), time_step)
            self.episode_lengths.clear()
            self.episode_rewards.clear()
            
    def _terminated_exp(self, experiences: List[Experience]) -> int:
        for i, exp in enumerate(experiences):
            if exp.terminated:
                return i
        return -1
    
    def _reset_total_rewards(self):
        self.total_rewards = [0] * self.clock.num_envs
