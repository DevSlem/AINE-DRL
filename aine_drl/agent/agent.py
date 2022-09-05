from typing import List
from aine_drl.trajectory import Trajectory
from aine_drl.drl_algorithm import DRLAlgorithm
from aine_drl.policy import Policy
from aine_drl.util import aine_api
from aine_drl.drl_util import Clock, Experience
import numpy as np
import torch

class Agent:
    def __init__(self, 
                 drl_algorithm: DRLAlgorithm, 
                 policy: Policy, 
                 trajectory: Trajectory,
                 clock: Clock) -> None:
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
        
    @aine_api
    def update(self, experience: List[Experience]):
        """
        Update the agent. It stores data, trains the DRL algorithm, etc.

        Args:
            experience (List[Experience]): the number of experiences must be the same as the number of environments.
        """
        # set trajectory
        self.trajectory.add(experience)
        # set clock
        self.clock.tick_time_step()
        if experience[0].terminated:
            self.clock.tick_episode()
        # if can log data
        if self.clock.check_time_step_freq:
            self.drl_algorithm.log_data(self.clock.time_step)
            self.policy.log_data(self.clock.time_step)
        # try training algorithm
        if self._try_train_algorithm():
            self.drl_algorithm.update_hyperparams(self.clock.time_step)
            self.policy.update_hyperparams(self.clock.time_step)
    
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
        return dist.sample()
    
    def _try_train_algorithm(self) -> bool:
        can_train = self.trajectory.can_train
        while self.trajectory.can_train:
            batch = self.trajectory.sample()
            # train the algorithm
            self.drl_algorithm.train(batch)
            self.clock.tick_training_step()
        return can_train
