from abc import ABC, abstractmethod
import time
from typing import Dict
from aine_drl.util import check_freq

class Clock:
    """
    Time utility.
    """
    
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.reset()
    
    def reset(self):
        self._global_time_step = 0
        self._episode = 0
        self._episode_len = 0
        self._real_start_time = time.time()
        self._real_time = 0.0
        self._training_step = 0
        
    @property
    def global_time_step(self) -> int:
        return self._global_time_step
    
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
    def real_time(self) -> float:
        return self._real_time
    
    def tick_gloabl_time_step(self):
        self._episode_len += 1
        self._global_time_step += self.num_envs
        self._real_time = self._get_real_time()
        
    def tick_episode(self):
        self._episode_len = 0
        self._episode += 1
        
    def tick_training_step(self):
        self._training_step += 1
        
    def check_global_time_step_freq(self, frequency: int) -> bool:
        """
        Check if the global time step is reached to the frequency. It considers multiple environments.
        """
        return check_freq(self._global_time_step, frequency, self.num_envs)
        
    def _get_real_time(self):
        return time.time() - self._real_start_time
    
    @property
    def state_dict(self) -> dict:
        clock_state_dict = {
            "clock": {
                "global_time_step": self._global_time_step,
                "episode": self._episode,
                "episode_len": self._episode_len,
                "training_step": self._training_step,
            }
        }
        return clock_state_dict
    
    def load_state_dict(self, state_dict: dict):
        state_dict = state_dict["clock"]
        for key, value in state_dict.items():
            setattr(self, f"_{key}", value)

class IClockNeed(ABC):
    @abstractmethod
    def set_clock(self, clock: Clock):
        """
        Set clock when Agent is instantiated.
        """
        raise NotImplementedError

class ILogable(IClockNeed):
    @property
    @abstractmethod
    def log_keys(self) -> tuple[str, ...]:
        """Returns log data keys."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def log_data(self) -> Dict[str, tuple]:
        """
        Returns log data and reset it.

        Returns:
            Dict[str, tuple]: key: (value, time)
        """
        raise NotImplementedError
