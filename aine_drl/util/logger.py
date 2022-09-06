from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
import aine_drl
import aine_drl.util as util

class Logger(ABC):
    @abstractmethod
    def log(self, key, value, t):
        """
        Log key-value pair at time t.
        """
        raise NotImplementedError
    
    def close(self):
        pass
    
class TensorBoardLogger(Logger):
    def __init__(self, logger: SummaryWriter) -> None:
        self.logger = logger
        
    def log(self, key, value, t):
        self.logger.add_scalar(key, value, t)
    
    def close(self):
        self.logger.flush()
        self.logger.close()
        
    @staticmethod
    def get_log_dir():
        dir = f"results/{aine_drl.get_env_id()}"
        return dir if not util.exists_dir(dir) else util.add_dir_num_suffix(dir, num_left=" (", num_right=")")

_logger: Logger = None
    
def set_logger(logger: Logger = None):
    global _logger
    if logger is None:
        logger = TensorBoardLogger(SummaryWriter(TensorBoardLogger.get_log_dir()))
    _logger = logger
    
def get_logger() -> Logger:
    global _logger
    return _logger
    
def log_data(key, value, t):
    global _logger
    assert _logger is not None, "You must call set_logger() method before call it."
    _logger.log(key, value, t)
    
def close_logger():
    global _logger
    if _logger is not None:
        _logger.close()
        _logger = None
    