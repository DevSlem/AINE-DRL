from typing import Union, List
from aine_drl import aine_api
from aine_drl.experience import Experience, Trajectory
import numpy as np

class BatchTrajectory(Trajectory):
    """
    It's a batch trajectory abstract class for the batch learning. It has the maximum experience count.
    """
    def __init__(self, max_count: int, env_count: int = 1) -> None:
        """
        Args:
            max_count (int): maximum count of experiences stored
            env_count (int, optional): environment count. Defaults to 1.
        """
        assert max_count > 0 and env_count > 0
        self.env_count = env_count
        self.max_count = max_count # maximum element count of flattend array
        self.reset()
        
    @aine_api
    @property
    def count(self) -> int:
        return self._count
        
    @aine_api
    def reset(self):
        self._count = 0 # total experience count of all environments
        self.recent_idx = -1 # index of the most recent experience
        
        self.states = [None] * self.max_count
        self.actions = [None] * self.max_count
        self.rewards = [None] * self.max_count
        self.terminateds = [None] * self.max_count
        self.next_state_buffer = [None] * self.env_count # most recently added next state
        
    @aine_api
    def add(self, experiences: Union[Experience, List[Experience]]):
        if isinstance(experiences, Experience):
            experiences = [experiences]
        
        for i, ex in enumerate(experiences):
            self.recent_idx = (self.recent_idx + 1) % self.max_count
            self._count = min(self._count + 1, self.max_count)
            
            self.states[self.recent_idx] = ex.state
            self.actions[self.recent_idx] = ex.action
            self.rewards[self.recent_idx] = ex.reward
            self.terminateds[self.recent_idx] = ex.terminated
            self.next_state_buffer[self.recent_idx % self.env_count] = ex.next_state
    