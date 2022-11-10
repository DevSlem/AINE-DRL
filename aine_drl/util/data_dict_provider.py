from abc import ABC
from typing import Tuple, Dict

class DataDictProvider(ABC):
    """
    log, state data dictionary provider.
    """
    
    @property
    def log_keys(self) -> Tuple[str, ...]:
        """Returns log data keys."""
        return tuple()
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        """
        Returns log data and reset it.

        Returns:
            Dict[str, tuple]: key: (value, time)
        """
        return {}
    
    @property
    def state_dict(self) -> dict:
        """Returns the state dict of the policy."""
        return {}
    
    def load_state_dict(self, state_dict: dict):
        """Load the state dict."""
        pass
