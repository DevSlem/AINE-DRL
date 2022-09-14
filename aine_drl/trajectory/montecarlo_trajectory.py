from aine_drl.drl_util import Experience, ExperienceBatch
from aine_drl.trajectory import Trajectory
from typing import List
from aine_drl import aine_api
import numpy as np

class MonteCarloTrajectory(Trajectory):
    """
    It's a trajectory for on-policy Monte Carlo methods.
    It samples whole trajectories when the episode is terminated, so the returned trajectory is the episode of the environment.
    """
    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.reset()
        
    @aine_api
    @property
    def count(self) -> int:
        return self._count
    
    @aine_api
    @property
    def can_train(self) -> bool:
        # if all episodes are terminated
        return np.array(self._can_train).sum() == self.num_envs
        
    @aine_api
    def reset(self):
        self._count = 0 # total experience count
        self._can_train = [False] * self.num_envs
        
        # shape: (num_envs, episode length)
        self.states = [[] for _ in range(self.num_envs)]
        self.actions = [[] for _ in range(self.num_envs)]
        self.rewards = [[] for _ in range(self.num_envs)]
        self.terminateds = [[] for _ in range(self.num_envs)]
        self.next_state_buffer = [None] * self.num_envs
        
    @aine_api
    def add(self, experiences: List[Experience]):
        assert len(experiences) == self.num_envs
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
        Returns the concatnated batch of multiple episodes.
        """
        for i in range(self.num_envs):
            self.states[i].append(self.next_state_buffer[i])
        
        states = []
        actions = []
        next_states = []
        rewards = []
        terminateds = []
        for i in range(self.num_envs):
            states.append(np.array(self.states[i][:-1]))
            actions.append(np.array(self.actions[i]))
            next_states.append(np.array(self.states[i][1:]))
            rewards.append(np.array(self.rewards[i]))
            terminateds.append(np.array(self.terminateds[i]))
        
        experience_batch = ExperienceBatch(
            np.concatenate(states),
            np.concatenate(actions),
            np.concatenate(next_states),
            np.concatenate(rewards),
            np.concatenate(terminateds)
        )
        
        self.reset()
        return experience_batch
    