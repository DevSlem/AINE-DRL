from aine_drl.experience import Experience, ExperienceBatch, Trajectory
from typing import Union, List
from aine_drl import aine_api
import numpy as np

class MonteCarloTrajectory(Trajectory):
    """
    It's a simple trajectory for on-policy Monte Carlo methods. It allows only one environment.
    """
    def __init__(self) -> None:
        self.reset()
        
    @aine_api
    @property
    def count(self) -> int:
        return self._count
    
    @aine_api
    @property
    def can_train(self) -> bool:
        return self._can_train
        
    @aine_api
    def reset(self):
        self._count = 0 # total experience count
        self._can_train = False
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminateds = []
        self.next_state_buffer = None
        
    @aine_api
    def add(self, experiences: Union[Experience, List[Experience]]):
        if isinstance(experiences, Experience):
            experiences = [experiences]
            
        # it allows only one environment
        assert len(experiences) == 1
        ex = experiences[0]
        
        self._count += 1
        self._can_train = ex.terminated
        
        self.states.append(ex.state)
        self.actions.append(ex.action)
        self.rewards.append(ex.reward)
        self.terminateds.append(ex.terminated)
        self.next_state_buffer = ex.next_state
            
    @aine_api
    def sample(self) -> ExperienceBatch:
        # if the trajectory has been terminated, then sample an experience batch (i.e. whole trajectory) and return it.
        if self._can_train:
            self.states.append(self.next_state_buffer)
            experience_batch = ExperienceBatch(
                np.array(self.states[:-1]), # states
                np.array(self.actions),
                np.array(self.states[1:]), # next states
                np.array(self.rewards),
                np.array(self.terminateds)
            )
            # because it's on-policy method, all experiences which are used to train must be discarded.
            self.reset()
            return experience_batch
            
        return None
    