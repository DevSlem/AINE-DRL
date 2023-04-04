from __future__ import annotations
import warnings

warnings.filterwarnings(action="ignore")
from torch.utils.tensorboard.writer import SummaryWriter

warnings.filterwarnings(action="default")

import builtins
from dataclasses import dataclass
from io import TextIOWrapper

import torch
import yaml

import aine_drl.util.func as util_f


class logger:
    """AINE-DRL logger class."""
    @dataclass(frozen=True)
    class __log_file:
        tb_logger: SummaryWriter
        log_message_file: TextIOWrapper
        
        def close(self):
            self.tb_logger.flush()
            self.tb_logger.close()
            self.log_message_file.close()
    
    _enabled = False
    _LOG_BASE_DIR: str = "results"
    _log_dir: str | None = None
    _log_file: __log_file | None = None
    
    @classmethod
    def enabled(cls) -> bool:
        return cls._enabled
    
    @classmethod
    def enable(cls, id: str, enable_log_file: bool = True):
        """Enalbe the logger."""
        if not cls._enabled:
            cls._enabled = True
            cls._log_dir = f"{cls._LOG_BASE_DIR}/{id}"
            if enable_log_file:
                cls._log_file = logger.__log_file(
                    tb_logger=SummaryWriter(cls._log_dir),
                    log_message_file=open(f"{cls._log_dir}/log_msg.txt", "a")
                )
        else:
            raise Exception("logger is already enabled")
    
    @classmethod
    def disable(cls):
        if cls._enabled:
            if cls._log_file is not None:
                cls._log_file.log_message_file.write("\n")
                cls._log_file.close()
                cls._log_file = None   
            cls._log_dir = None
            cls._enabled = False
        else:
            raise Exception("you can only disable when the logger is enabled")
    
    @classmethod    
    def print(cls, message: str, prefix: str = "[AINE-DRL] "):
        """Print message with prefix."""
        builtins.print(f"{prefix}{message}")
        if cls._log_file is not None:
            cls._log_file.log_message_file.write(f"{prefix}{message}\n")
    
    @classmethod
    def numbering_env_id(cls, env_id: str) -> str:
        """Numbering environment id if it already exsits."""
        dir = f"{cls._LOG_BASE_DIR}/{env_id}"
        if util_f.exists_dir(dir):
            dir = util_f.add_dir_num_suffix(dir, num_left="_")
            env_id = dir[len(cls._LOG_BASE_DIR) + 1:]
        return env_id
    
    @classmethod
    def log_base_dir(cls) -> str:
        """Returns the log base directory."""
        return cls._LOG_BASE_DIR
    
    @classmethod
    def log_dir(cls) -> str | None:
        """Returns the log directory."""
        return cls._log_dir
    
    @classmethod
    def agent_save_path(cls) -> str:
        """Returns agent save directory."""
        if cls._log_dir is None:
            raise Exception("you must enable the logger")
        return f"{cls._log_dir}/agent.pt"
        
    @classmethod
    def log(cls, key, value, t):
        """Records the log."""
        if cls._log_file is None:
            raise Exception("you need to enable the logger with enable_log_file option")
        cls._log_file.tb_logger.add_scalar(key, value, t)
        
    @classmethod
    def load_agent(cls, path: str | None = None) -> dict:
        return torch.load(cls.agent_save_path() if path is None else path)
    
    @classmethod
    def save_agent(cls, state_dict: dict, time_steps: int | None = None):
        torch.save(state_dict, cls.agent_save_path())
        if time_steps is not None:
            specific_agent_dir = f"{cls.log_dir()}/agents"
            util_f.create_dir(specific_agent_dir)
            torch.save(state_dict, f"{specific_agent_dir}/agent_{time_steps}.pt")
            
    @classmethod
    def save_config_dict_to_yaml(cls, config: dict):
        if cls._log_dir is None:
            raise Exception("you must enable the logger")
        util_f.create_dir(cls._log_dir)
        with open(f"{cls._log_dir}/config.yaml", "w") as f:
            yaml.dump(config, f)

class TextInfoBox:
    def __init__(self, right_margin: int = 10) -> None:
        self._texts = []
        self._right_margin = right_margin
        self._max_text_len = 0
        
    def add_text(self, text: str | None) -> "TextInfoBox":
        if text is None:
            return self
        self._max_text_len = max(self._max_text_len, len(text))
        self._texts.append((f" {text} ", " "))
        return self
        
    def add_line(self, marker: str = "-") -> "TextInfoBox":
        if len(marker) != 1:
            raise ValueError(f"marker must be one character, but {marker}")
        self._texts.append(("", marker))
        return self
    
    def make(self) -> str:
        text_info_box = f"+{self._horizontal_line()}+\n"
        for text, marker in self._texts:
            text_info_box += f"|{text}{marker * (self._max_space_len - len(text))}|\n"
        text_info_box += f"+{self._horizontal_line()}+"
        return text_info_box
    
    def _horizontal_line(self, marker: str = "-") -> str:
        return marker * (self._max_space_len)

    @property
    def _max_space_len(self) -> int:
        return self._max_text_len + self._right_margin