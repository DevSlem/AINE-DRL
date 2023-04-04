from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Iterable

import yaml

import aine_drl.util.func as util_f
from aine_drl.agent.agent import Agent
from aine_drl.env import Env, GymEnv, GymRenderableEnv, MLAgentsEnv
from aine_drl.run.inference import Inference, InferenceConfig
from aine_drl.run.train import Train, TrainConfig
from aine_drl.util.logger import logger


class AgentFactory(ABC):
    """When you use `AINEFactory`, you need to implement this class."""
    @abstractmethod
    def make(self, env: Env, config_dict: dict) -> Agent:
        raise NotImplementedError

class AINEFactoryError(KeyError):
    pass

T = TypeVar("T")

class AINEFactory(Generic[T]):
    def __init__(self, config: dict) -> None:
        self._id = tuple(config.keys())[0]
        try:
            self._config_dict: dict = config[self._id]
        except KeyError:
            raise AINEFactoryError("Invalid config")
        
        self._env: Env | None = None
        self._agent = None
        
        if not logger.enabled():
            logger.enable(self._id, enable_log_file=False)
        logger.save_config_dict_to_yaml(config)
        logger.disable()
    
    @abstractmethod
    def make_env(self) -> "AINEFactory[T]":
        """Make an environment from the configuration."""
        raise NotImplementedError
    
    def set_env(self, env: Env) -> "AINEFactory[T]":
        """Set an environment manually."""
        if self._env is not None:
            raise AINEFactoryError("environment is already set")
        self._env = env
        return self
    
    def make_agent(self, agent_factory: AgentFactory) -> "AINEFactory[T]":
        """Make an agent from the configuration."""
        if self._env is None:
            raise AINEFactoryError("you need to make or set environment first")
        self._agent = agent_factory.make(self._env, self._config_dict.get("Agent", dict()))
        return self
    
    @abstractmethod
    def ready(self) -> T:
        """Get ready to train or inference."""
        raise NotImplementedError
        
    @property
    def id(self) -> str:
        """Get the id of the configuration."""
        return self._id
    
    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Get the number of environments from the configuration."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def seed(self) -> int | list[int] | None:
        """Get the seed from the configuration."""
        raise NotImplementedError

class AINETrainFactory(AINEFactory[Train]):
    """
    A factory class to make `Train` instance from a configuration.
    
    Refer to the following steps:
    
    1. Make a `AINETrainFactory` instance from a configuration file.
    2. Make or set an environment.
    3. Make an agent.
    4. Get ready to train.
    5. Start training.
    6. Close the process.
    
    Example::

        AINETrainFactory.from_yaml("config.yaml") \\
            .make_env() \\
            .make_agent(AgentFactory()) \\
            .ready() \\
            .train() \\
            .close()
    """
    def make_env(self) -> "AINETrainFactory":
        if self._env is not None:
            raise AINEFactoryError("environment is already set")
        env_dict: dict = self._config_dict["Env"]
        seed = self.seed
        if isinstance(seed, int):
            util_f.seed(seed)
        elif isinstance(seed, Iterable) and len(seed) > 0:
            util_f.seed(seed[0])
        self._env = self._make_train_env(env_dict, self.num_envs, seed)
        return self
    
    def set_env(self, env: Env) -> "AINEFactory[Train]":
        if env.num_envs != self.num_envs:
            warnings.warn("the number of environments is different from the configuration. this may cause an unexpected behavior.")
        return super().set_env(env)
    
    def _make_train_env(self, env_dict: dict, num_envs: int, seed: int | list[int] | None) -> Env:        
        config_dict: dict = env_dict["Config"]
        if "seed" not in config_dict.keys():
            config_dict["seed"] = seed
        env_type = env_dict["type"]
        if env_type == "Gym":
            return GymEnv.from_gym_make(num_envs=num_envs, **config_dict)
        elif env_type == "ML-Agents":
            if "id" in config_dict.keys():
                return MLAgentsEnv.from_registry(num_envs=num_envs, **config_dict)
            elif "file_name" in config_dict.keys():
                return MLAgentsEnv.from_unity_env(num_envs=num_envs, **config_dict)
            else:
                raise AINEFactoryError("you must specify either `id` or `file_name` in the configuration")
        else:
            raise AINEFactoryError("invalid environment type")
    
    def ready(self) -> Train:
        if self._env is None:
            raise AINEFactoryError("you need to make or set environment first")
        if self._agent is None:
            raise AINEFactoryError("you need to make agent first")
        train_dict: dict = self._config_dict["Train"]
        train_config = TrainConfig(**train_dict["Config"])
        return Train(self.id, train_config, self._env, self._agent)
    
    @property
    def num_envs(self) -> int:
        return self._config_dict["Train"].get("num_envs", 1)
    
    @property
    def seed(self) -> int | list[int] | None:
        return self._config_dict["Train"].get("seed", None)
    
    @staticmethod
    def from_yaml(file_path: str) -> "AINETrainFactory":
        with open(file_path) as f:
            config = yaml.load(f, yaml.FullLoader)
        return AINETrainFactory(config)
    
class AINEInferenceFactory(AINEFactory[Inference]):
    """
    A factory class to make `Inference` instance from a configuration.
    
    Refer to the following steps:
    
    1. Make a `AINEInferenceFactory` instance from a configuration file.
    2. Make or set an environment.
    3. Make an agent.
    4. Get ready to inference.
    5. Start inference.
    6. Close the process.
    
    Example::

        AINEInferenceFactory.from_yaml("config.yaml") \\
            .make_env() \\
            .make_agent(AgentFactory()) \\
            .ready() \\
            .inference() \\
            .close()
    """
    def make_env(self) -> "AINEInferenceFactory":
        if self._env is not None:
            raise AINEFactoryError("environment is already set")
        env_dict: dict = self._config_dict["Env"]
        inference_dict: dict = self._config_dict["Inference"]
        seed = self.seed
        if isinstance(seed, int):
            util_f.seed(seed)
        elif isinstance(seed, Iterable) and len(seed) > 0:
            util_f.seed(seed[0])
        inference_config = InferenceConfig(**inference_dict["Config"])
        self._env = self._make_inference_env(env_dict, inference_config.export, seed)
        return self
    
    def ready(self) -> Inference:
        if self._env is None:
            raise AINEFactoryError("you need to make or set environment first")
        if self._agent is None:
            raise AINEFactoryError("you need to make agent first")
        inference_dict: dict = self._config_dict["Inference"]
        inference_config = InferenceConfig(**inference_dict["Config"])
        return Inference(self.id, inference_config, self._env, self._agent)
    
    def _make_inference_env(self, env_dict: dict, export: str | None, seed: int | list[int] | None) -> Env:
        config_dict: dict = env_dict["Config"]
        env_type = env_dict["type"]
        if env_type == "Gym":
            if export is None:
                render_mode = None
            elif export == "render_only":
                render_mode = "human"
            else:
                render_mode = "rgb_array"
            config_dict["render_mode"] = render_mode
            if "seed" not in config_dict.keys():
                config_dict["seed"] = seed
            return GymRenderableEnv.from_gym_make(**config_dict)
        elif env_type == "ML-Agents":
            raise NotImplementedError("ML-Agents environment for inference is not implemented yet")
        else:
            raise AINEFactoryError("invalid environment type")
            
    @property
    def num_envs(self) -> int:
        return 1

    @property
    def seed(self) -> int | list[int] | None:
        return self._config_dict["Inference"].get("seed", None)
            
    @staticmethod
    def from_yaml(file_path: str) -> "AINEInferenceFactory":
        with open(file_path) as f:
            config = yaml.load(f, yaml.FullLoader)
        return AINEInferenceFactory(config)