from abc import ABC, abstractmethod
import numpy as np

class Decay(ABC):
    """
    Decay function utility.
    """
    
    @abstractmethod
    def value(self, t: float) -> float:
        """Returns decay function output for t.

        Args:
            t (float): time

        Returns:
            float: decay function output
        """
        raise NotImplementedError
    
    def __call__(self, t: float) -> float:
        return self.value(t)
    
class NoDecay(Decay):
    """
    Just returns fixed value.
    """
    def __init__(self, value: float) -> None:
        self.val = value
        
    def value(self, t: float) -> float:
        return self.val
    
class LinearDecay(Decay):
    """
    Linear decay.
    """ 
    def __init__(self, start_value: float, end_value: float, start_t: int, end_t: int) -> None:
        assert start_t < end_t and start_value >= end_value
        self.start_val = start_value
        self.end_val = end_value
        self.start_t = start_t
        self.end_t = end_t
        
    def value(self, t: float) -> float:
        slope = (self.end_val - self.start_val) / (self.end_t - self.start_t)
        val = np.clip(slope * (t - self.start_t) + self.start_val, self.end_val, self.start_val)
        return val
