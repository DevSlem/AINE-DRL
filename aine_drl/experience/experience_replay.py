from typing import List
from aine_drl.experience import BatchTrajectory, Experience, ExperienceBatch
import numpy as np
import aine_drl.util as util
from aine_drl.util.decorator import aine_api


class ExperienceReplay(BatchTrajectory):
    """
    It's a simple experience replay memory which is usually used for DQN.
    It samples experiences randomly.
    """
    def __init__(self, 
                 training_freq: int, 
                 batch_size: int, 
                 max_count: int, 
                 env_count: int = 1,
                 epoch: int = 1) -> None:
        assert training_freq > 0 and batch_size > 0 and epoch > 0
        super().__init__(max_count, env_count)
        self.freq = training_freq
        self.batch_size = batch_size
        self.epoch = epoch
        
    @aine_api
    @property
    def can_train(self) -> bool:
        return self.added_ex_count >= self.freq and self.current_epoch < self.epoch
    
    @aine_api
    def reset(self):
        super().reset()
        self.added_ex_count = 0
        self.current_epoch = 0
        
    @aine_api
    def add(self, experiences: List[Experience]):
        super().add(experiences)
        self.added_ex_count += len(experiences)
    
    @aine_api
    def sample(self) -> ExperienceBatch:
        self.current_epoch += 1
        # if the current epoch is the last one
        if self.current_epoch >= self.epoch:
            self.added_ex_count -= self.freq
            self.current_epoch = 0
            
        batch_idxs = self._sample_idxs()
        experience_batch = ExperienceBatch(
            util.get_batch(self.states, batch_idxs),
            util.get_batch(self.actions, batch_idxs),
            self._sample_next_states(batch_idxs),
            util.get_batch(self.rewards, batch_idxs),
            util.get_batch(self.terminateds, batch_idxs)
        )
        return experience_batch
        
    def _sample_idxs(self) -> np.ndarray:
        batch_idxs = np.random.randint(self._count, size=self.batch_size)
        return batch_idxs
        
    def _sample_next_states(self, batch_idxs: np.ndarray) -> np.ndarray:
        """
        Sample next states from the trajectory. TODO: #6 It needs to be tested.
        
        The source of this method is kengz/SLM-Lab (Github) https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/replay.py.

        Args:
            batch_idxs (np.ndarray): batch indexes which mean current state indexes

        Returns:
            np.ndarray: next state batch
        """
        # [state1, state2, state3, next_state1, next_state2, next_state3]
        next_state_idxs = (batch_idxs + self.env_count) % self.max_count
        # if recent < next_state_index <= recent + env_count, next_state is stored in next_state_buffer
        # it has two cases
        # case 1) [recent1, recent2, recent3, oldest1, oldest2, oldest3]
        # if batch_index is 1 (recent2), then 2 (recent=recent3) < 1+3 (next_state_index=oldest2) <= 2+3
        # case 2) [prev1, prev2, prev3, recent1, recent2, recent3]
        # if batch_index is 4 (recent2), then 5 (recent=recent3) < 4+3 (next_state_index not exists) < 5+3
        not_exsists_next_state = np.argwhere(
            (self.recent_idx < next_state_idxs) & (next_state_idxs <= self.recent_idx + self.env_count)
        ).flatten()
        # check if there is any indexes to get from buffer
        do_replace = not_exsists_next_state.size != 0
        if do_replace:
            # to avoid index out of range exception because of case 2
            next_state_idxs[not_exsists_next_state] = 0
        # get the next state batch
        next_states = util.get_batch(self.states, next_state_idxs)
        if do_replace:
            # recent < next_state_index <= recent + env_count
            # 0 <= next_state_index - recent - 1 < env_count
            next_state_buffer_idxs = next_state_idxs[not_exsists_next_state] - self.recent_idx - 1
            # replace them
            next_states[not_exsists_next_state] = util.get_batch(self.next_state_buffer, next_state_buffer_idxs)
        return next_states