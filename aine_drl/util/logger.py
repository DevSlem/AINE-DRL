from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter

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
        self.logger.close()

class GlobalLogger:
    logger: Logger = None
    
def set_logger(logger: Logger):
    GlobalLogger.logger = logger
    
def log(key, value, t):
    GlobalLogger.logger.log(key, value, t)
    
def close_logger():
    GlobalLogger.logger.close()
    GlobalLogger.logger = None
