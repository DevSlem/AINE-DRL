import time
from aine_drl.util import check_freq

class Clock:
    """
    Don't use it yet. It needs to be implemented.
    """
    
    def __init__(self, num_env: int, frequency: int) -> None:
        self.num_env = num_env
        self.frequency = frequency
        self.reset()
    
    def reset(self):
        self._time_step = 0
        self._episode = 0
        self._episode_len = 0
        self._real_start_time = time.time()
        self._real_time = 0.0
        self._training_step = 0
        
    @property
    def time_step(self) -> int:
        return self._time_step
    
    @property
    def episode(self) -> int:
        return self._episode
    
    @property
    def episode_len(self) -> int:
        return self._episode_len
    
    @property
    def training_step(self) -> int:
        return self._training_step
    
    @property
    def real_time(self) -> int:
        return int(self._real_time)
    
    @property
    def check_time_step_freq(self) -> bool:
        """
        Check if the time step is reached to the frequency. It considers multiple environments.
        """
        return check_freq(self.time_step, self.frequency, self.num_env)
    
    def tick_time_step(self):
        self._episode_len += 1
        self._time_step += self.num_env
        self._real_time = self._get_real_time()
        
    def tick_episode(self):
        self._episode_len = 0
        self._episode += 1
        
    def tick_training_step(self):
        self._training_step += 1
        
    def _get_real_time(self):
        return time.time() - self._real_start_time
