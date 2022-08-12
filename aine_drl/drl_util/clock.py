import time

class Clock:
    """
    Don't use it yet. It needs to be implemented.
    """
    
    def __init__(self) -> None:
        raise NotImplementedError
        self.reset()
    
    def reset(self):
        self._time_step = 0
        self._episode = 0
        self._total_time_step = 0
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
    def total_time_step(self) -> int:
        return self._total_time_step
    
    @property
    def training_step(self) -> int:
        return self._training_step
    
    @property
    def real_time(self) -> int:
        return int(self._real_time)
    
    def tick_time_step(self):
        self._time_step += 1
        self._total_time_step += 1
        self._real_time = self._get_real_time()
        
    def tick_episode(self):
        self._time_step = 0
        self._episode += 1
        
    def tick_training_step(self):
        self._training_step += 1
        
    def _get_real_time(self):
        return time.time() - self._real_start_time
        