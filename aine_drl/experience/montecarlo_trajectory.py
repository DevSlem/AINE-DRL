from aine_drl.experience import Experience, ExperienceBatch, Trajectory
from typing import List
from aine_drl import aine_api
import numpy as np

class MonteCarloTrajectory(Trajectory):
    """
    It's a trajectory for on-policy Monte Carlo methods.
    It samples whole trajectories when the episode is terminated, so the returned trajectory is the episode of the environment.
    """
    def __init__(self, env_count: int = 1) -> None:
        self.reset()
        self.env_count = env_count
        
    @aine_api
    @property
    def count(self) -> int:
        return self._count
    
    @aine_api
    @property
    def can_train(self) -> bool:
        # if all episodes are terminated
        return np.array(self._can_train).sum() == self.env_count
        
    @aine_api
    def reset(self):
        self._count = 0 # total experience count
        self._can_train = [False] * self.env_count
        self.returned_trajectory = 0
        
        # shape: (env_count, episode length)
        self.states = [[] for _ in range(self.env_count)]
        self.actions = [[] for _ in range(self.env_count)]
        self.rewards = [[] for _ in range(self.env_count)]
        self.terminateds = [[] for _ in range(self.env_count)]
        self.next_state_buffer = [None] * self.env_count
        
    @aine_api
    def add(self, experiences: List[Experience]):
        assert len(experiences) == self.env_count
        for i, ex in enumerate(experiences):
            # if the episode of the environment has been terminated, skip
            if self._can_train[i]:
                continue
            
            self._count += 1
            self._can_train[i] = ex.terminated
            
            self.states[i].append(ex.state)
            self.actions[i].append(ex.action)
            self.rewards[i].append(ex.reward)
            self.terminateds[i].append(ex.terminated)
            self.next_state_buffer[i] = ex.next_state
            
    @aine_api
    def sample(self) -> ExperienceBatch:
        """
        Returns each trajectory, that is an episode. You need to call it while can train.

        Returns:
            ExperienceBatch: an episode
        """
        i = self.returned_trajectory
        self.states[i].append(self.next_state_buffer)
        experience_batch = ExperienceBatch.create(
            self.states[i][:-1], # states
            self.actions[i],
            self.states[i][1:], # next states
            self.rewards[i],
            self.terminateds[i]
        )
        self.returned_trajectory += 1
        # if all trajectories has been returned, then reset
        if self.returned_trajectory == self.env_count:
            self.reset()
        return experience_batch
    