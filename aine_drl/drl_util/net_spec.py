from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class NetSpec(ABC):
    @property
    @abstractmethod
    def state_dict(self) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        raise NotImplementedError
