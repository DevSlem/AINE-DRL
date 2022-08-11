from abc import ABC
from typing import List
from aine_drl.experience import Trajectory, Experience
from aine_drl.policy import Policy
from aine_drl.util.decorator import aine_api
import numpy as np

class Agent(ABC):
    def __init__(self, 
                 drl_algorithm: DRLAlgorithm, 
                 policy: Policy, 
                 trajectory: Trajectory) -> None:
        self.drl_algorithm = drl_algorithm
        self.policy = policy
        self.trajectory = trajectory
        
    @aine_api
    def update(self, experience: List[Experience]):
        """
        Update the agent. It stores data, trains the DRL algorithm, etc.

        Args:
            experience (List[Experience]): experiences of which the element count must be the environment count.
        """
        self.trajectory.add(experience)
        if self._train_algorithm():
            self.policy.update_hyperparams(time_step)
    
    @aine_api
    def act(self, states: np.ndarray) -> np.ndarray:
        """
        Returns actions from the state batch

        Args:
            states (np.ndarray): state batch

        Returns:
            np.ndarray: actions
        """
        pass
    
    def _train_algorithm(self) -> bool:
        can_train = self.trajectory.can_train
        while self.trajectory.can_train:
            batch = self.trajectory.sample()
            # train the algorithm
            # self.drl_algorithm.train(batch)
        return can_train
