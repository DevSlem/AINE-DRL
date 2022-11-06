import warnings
warnings.filterwarnings(action="ignore")
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings(action="default")
import aine_drl
import aine_drl.util as util
import builtins

class logger:
    """ Standard logger class. """
    
    _log_base_dir = "results"
    _log_dir = None
    _logger: SummaryWriter = None
    
    @staticmethod    
    def print(message: str, prefix: str = "[AINE-DRL]"):
        """Print message with prefix."""
        builtins.print(f"{prefix} {message}")
    
    @classmethod
    def numbering_env_id(cls, env_id: str) -> str:
        """Numbering environment id if it already exsits."""
        dir = f"{cls._log_base_dir}/{env_id}"
        if util.exists_dir(dir):
            dir = util.add_dir_num_suffix(dir, num_left="_")
            env_id = dir[len(cls._log_base_dir) + 1:]
        return env_id
    
    @classmethod
    def log_base_dir(cls) -> str:
        """Returns the log base directory."""
        return cls._log_base_dir
    
    @classmethod
    def log_dir(cls) -> str:
        """Returns the log directory."""
        if cls._log_dir is None:
            raise Exception("You must call logger.start() method first before call this method.")
        return cls._log_dir
    
    @classmethod
    def start(cls, env_id: str):
        """Set logging setting."""
        if cls._log_dir is None:
            cls._log_dir = f"{cls._log_base_dir}/{env_id}"
            cls._logger = SummaryWriter(cls._log_dir)
        else:
            raise Exception("logger is currently progressing. You should call it after ending the current logger using logger.end() method.")
    
    @classmethod
    def agent_save_dir(cls):
        """Returns agent save directory."""
        if cls._log_dir is None:
            raise Exception("You must call logger.start() method first before call this method.")
        return f"{cls._log_dir}/agent.pt"
        
    @classmethod
    def log(cls, key, value, t):
        """Records the log."""
        if cls._logger is None:
            raise Exception("You must call logger.start() method first before call this method.")
        cls._logger.add_scalar(key, value, t)
        
    @classmethod
    def end(cls):
        if cls._logger is not None:
            cls._logger.flush()
            cls._logger.close()
            
            cls._log_dir = None
            cls._logger = None
        else:
            raise Exception("You can only call this method when the logger is progressing.")
            
    # @classmethod
    # def log_lr_scheduler(cls, lr_scheduler, t, key = "Learning Rate"):
    #     if lr_scheduler is not None:
    #         lr = lr_scheduler.get_lr()
    #         cls.log(f"Network/{key}", lr if type(lr) is float else lr[0], t)
