import time

class Clock:
    """
    Don't use it yet. It needs to be implemented.
    """
    
    def __init__(self, env_count: int) -> None:
        self.env_count = env_count
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
    
    def tick_time_step(self):
        self._episode_len += 1
        self._time_step += self.env_count
        self._real_time = self._get_real_time()
        
    def tick_episode(self):
        self._episode_len = 0
        self._episode += 1
        
    def tick_training_step(self):
        self._training_step += 1
        
    def _get_real_time(self):
        return time.time() - self._real_start_time
        