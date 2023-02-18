import warnings
warnings.filterwarnings(action="ignore")
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings(action="default")
import aine_drl.util as util
import builtins
import torch

class logger:
    """ Standard logger class. """
    
    _log_base_dir = "results"
    _log_dir = None
    _logger: SummaryWriter = None
    
    @staticmethod    
    def print(message: str, prefix: str = "[AINE-DRL] "):
        """Print message with prefix."""
        builtins.print(f"{prefix}{message}")
    
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
            raise Exception("You must have initialized log directory.")
        return cls._log_dir
    
    @classmethod
    def set_log_dir(cls, env_id: str):
        if cls._logger is None:
            cls._log_dir = f"{cls._log_base_dir}/{env_id}"
        else:
            raise Exception("logger is currently working. You must have called it after ending the current logger using logger.end() method.")
    
    @classmethod
    def start(cls, env_id: str, ):
        """Set logging setting."""
        if cls._logger is None:
            cls.set_log_dir(env_id)
            cls._logger = SummaryWriter(cls._log_dir)
        else:
            raise Exception("logger is currently working. You must have called it after ending the current logger using logger.end() method.")
    
    @classmethod
    def agent_save_dir(cls):
        """Returns agent save directory."""
        if cls._log_dir is None:
            raise Exception("You must have initialized log directory.")
        return f"{cls._log_dir}/agent.pt"
        
    @classmethod
    def log(cls, key, value, t):
        """Records the log."""
        if cls._logger is None:
            raise Exception("You must call logger.start() method first before call this method.")
        cls._logger.add_scalar(key, value, t)
        
    @classmethod
    def load_agent(cls) -> dict:
        return torch.load(cls.agent_save_dir())
    
    @classmethod
    def save_agent(cls, state_dict: dict):
        torch.save(state_dict, cls.agent_save_dir())
        
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

class TextInfoBox:
    def __init__(self,
                 max_text_len: int) -> None:
        self._texts = []
        self._max_space_len = max_text_len + 2
        
    def add_text(self, text: str):
        if len(text) > self._max_space_len - 2:
            raise ValueError(f"text must be less than {self._max_space_len - 2} characters, but {len(text)}")
        self._texts.append(f" {text} ")
        
    def add_line(self, marker: str = "-"):
        if len(marker) != 1:
            raise ValueError(f"marker must be one character, but {marker}")
        self._texts.append(self._horizontal_line(marker))
    
    def make(self) -> str:
        text_info_box = f"+{self._horizontal_line()}+\n"
        for text in self._texts:
            text_info_box += f"|{text}{self._remained_whitespace(text)}|\n"
        text_info_box += f"+{self._horizontal_line()}+"
        return text_info_box
        
    def _remained_whitespace(self, text: str) -> str:
        return " " * (self._max_space_len - len(text))
    
    def _horizontal_line(self, marker: str = "-") -> str:
        return marker * (self._max_space_len)
