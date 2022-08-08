from aine_drl.experience import Experience, ExperienceBatch, Trajectory
from typing import Union, List
from aine_drl import aine_api
import numpy as np

class EpisodicTrajectory(Trajectory):
    """
    It's a trajectory for Monte Carlo methods.
    """
    def __init__(self, env_count: int = 1) -> None:
        assert env_count > 0
        self.env_count = env_count
        self.reset()
        
    @aine_api
    @property
    def count(self) -> int:
        return self._count
    
    @aine_api
    @property
    def can_train(self) -> bool:
        # if at least one trajectory has been terminated
        return np.array(self._can_train).sum() > 0
        
    @aine_api
    def reset(self):
        self._count = 0 # total experience count of all environments
        self._can_train = [False] * self.env_count
        
        # each list shape: (env count, trajectory length of each env)
        self.states = [[] for i in range(self.env_count)]
        self.actions = [[] for i in range(self.env_count)]
        self.rewards = [[] for i in range(self.env_count)]
        self.terminateds = [[] for i in range(self.env_count)]
        self.next_state_buffer = [None] * self.env_count # most recently added next state
        
    @aine_api
    def add(self, experiences: Union[Experience, List[Experience]]):
        if isinstance(experiences, Experience):
            experiences = [experiences]
            
        assert len(experiences) == self.env_count
        
        for i, ex in enumerate(experiences):
            self._count += 1
            self._can_train[i] = ex.terminated
            
            self.states[i].append(ex.state)
            self.actions[i].append(ex.action)
            self.rewards[i].append(ex.reward)
            self.terminateds[i].append(ex.terminated)
            self.next_state_buffer[i] = ex.next_state
            
    @aine_api
    def sample(self) -> ExperienceBatch:
        for i, can_train in enumerate(self._can_train):
            # if ith trajectory has been terminated, then immediately sample an experience batch (i.e. whole trajectory) and return it. 
            if can_train:
                self.states[i].append(self.next_state_buffer[i])
                experience_batch = ExperienceBatch(
                    np.array(self.states[i][:-1]),
                    np.array(self.actions[i]),
                    np.array(self.states[i][1:]),
                    np.array(self.rewards[i]),
                    np.array(self.rewards[i])
                )
                self._count -= len(self.actions[i])
                self._can_train[i] = False
                self.states[i].clear()
                self.actions[i].clear()
                self.rewards[i].clear()
                self.terminateds[i].clear()
                self.next_state_buffer[i] = None
                return experience_batch
            
        return None
    