import warnings
warnings.filterwarnings(action="ignore")
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings(action="default")
import aine_drl
import aine_drl.util as util
import builtins

class logger:
    """ Standard logger class. """
    
    @staticmethod
    def get_log_dir():
        """ Returns the log directory. """
        dir = f"results/{aine_drl.get_global_env_id()}"
        return dir if not util.exists_dir(dir) else util.add_dir_num_suffix(dir, num_left="_")
    
    _logger: SummaryWriter = None
        
    @classmethod
    def log(cls, key, value, t):
        """ Records a log using the logger. """
        if cls._logger is None:
            cls._logger = SummaryWriter(cls.get_log_dir())
        cls._logger.add_scalar(key, value, t)
        
    @classmethod
    def close(cls):
        if cls._logger is not None:
            cls._logger.flush()
            cls._logger.close()
            
    @classmethod
    def log_lr_scheduler(cls, lr_scheduler, t, key = "Learning Rate"):
        if lr_scheduler is not None:
            lr = lr_scheduler.get_lr()
            cls.log(f"Network/{key}", lr if type(lr) is float else lr[0], t)
    
    @staticmethod    
    def print(message: str, prefix: str = "[AINE-DRL]"):
        builtins.print(f"{prefix} {message}")
