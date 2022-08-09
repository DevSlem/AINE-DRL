from aine_drl.experience import BatchTrajectory, ExperienceBatch
from aine_drl.util.decorator import aine_api

class OnPolicyTrajectory(BatchTrajectory):
    """
    It's an on-policy trajectory for the batch learning.
    """    
    def __init__(self, training_freq_per_env: int, env_count: int = 1) -> None:
        """
        Args:
            training_freq_per_env (int): transition count per environment for training
            env_count (int, optional): environment count. Defaults to 1.
        """
        assert training_freq_per_env > 0
        super().__init__(training_freq_per_env * env_count, env_count)
        self.freq = training_freq_per_env
        
    @aine_api
    @property
    def can_train(self) -> bool:
        return int(self._count / self.env_count) == self.freq
    
    @aine_api
    def sample(self) -> ExperienceBatch:
        """
        Sample from the trajectory. You should call this function only if can train.
        
        The batch size is training_freq_per_env * env_count.

        Returns:
            ExperienceBatch: sampled experience batch
        """
        # temporarily extends
        self.states.extend(self.next_state_buffer)
        experience_batch = ExperienceBatch.create(
            self.states[:-self.env_count], # states
            self.actions,
            self.states[self.env_count:], # next states
            self.rewards,
            self.terminateds
        )
        # because it's on-policy method, all experiences which are used to train must be discarded.
        self.reset()
        return experience_batch
    