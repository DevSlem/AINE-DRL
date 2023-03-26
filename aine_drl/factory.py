import warnings
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import yaml

import aine_drl.util as util
from aine_drl.agent.agent import Agent
from aine_drl.train.env import Env, GymEnv, GymRenderableEnv
from aine_drl.train.inference import Inference, InferenceConfig
from aine_drl.train.train import Train, TrainConfig


class AgentFactory(ABC):
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
    
    @abstractmethod
    def make_env(self) -> "AINEFactory[T]":
        raise NotImplementedError
    
    def set_env(self, env: Env) -> "AINEFactory[T]":
        if self._env is not None:
            raise AINEFactoryError("environment is already set")
        self._env = env
        return self
    
    def make_agent(self, agent_factory: AgentFactory) -> "AINEFactory[T]":
        if self._env is None:
            raise AINEFactoryError("you need to make or set environment first")
        self._agent = agent_factory.make(self._env, self._config_dict.get("Agent", dict()))
        return self
    
    @abstractmethod
    def ready(self) -> T:
        raise NotImplementedError
        
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def num_envs(self) -> int:
        return self._config_dict["Train"].get("num_envs", 1)

class AINETrainFactory(AINEFactory[Train]):
    def make_env(self) -> "AINETrainFactory":
        if self._env is not None:
            raise AINEFactoryError("environment is already set")
        env_dict: dict = self._config_dict["Env"]
        train_dict: dict = self._config_dict["Train"]
        num_envs = train_dict.get("num_envs", 1)
        seed = train_dict.get("seed", None)
        if seed is not None:
            util.seed(seed)
        self._env = self._make_train_env(env_dict, num_envs, seed)
        return self
    
    def set_env(self, env: Env) -> "AINEFactory[Train]":
        if env.num_envs != self.num_envs:
            warnings.warn("the number of environments is different from the configuration. this may cause an unexpected behavior.")
        return super().set_env(env)
    
    def _make_train_env(self, env_dict: dict, num_envs: int, seed: int | None) -> Env:        
        config_dict: dict = env_dict["Config"]
        match env_dict["type"]:
            case "Gym":
                if "seed" not in config_dict.keys():
                    config_dict["seed"] = seed
                return GymEnv.from_gym_make(num_envs=num_envs, **config_dict)
            case "ML-Agents":
                raise NotImplementedError("ML-Agents environment is not implemented yet")
            case _:
                raise AINEFactoryError("invalid environment type")
    
    def ready(self) -> Train:
        if self._env is None:
            raise AINEFactoryError("you need to make or set environment first")
        if self._agent is None:
            raise AINEFactoryError("you need to make agent first")
        train_dict: dict = self._config_dict["Train"]
        train_config = TrainConfig(**train_dict["Config"])
        return Train(self.id, train_config, self._env, self._agent)
    
    @staticmethod
    def from_yaml(file_path: str) -> "AINETrainFactory":
        with open(file_path) as f:
            config = yaml.load(f, yaml.FullLoader)
        return AINETrainFactory(config)
    
class AINEInferenceFactory(AINEFactory[Inference]):
    def make_env(self) -> "AINEInferenceFactory":
        if self._env is not None:
            raise AINEFactoryError("environment is already set")
        env_dict: dict = self._config_dict["Env"]
        inference_dict: dict = self._config_dict["Inference"]
        seed = inference_dict.get("seed", None)
        if seed is not None:
            util.seed(seed)
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
    
    def _make_inference_env(self, env_dict: dict, export: str | None, seed: int | None) -> Env:
        config_dict: dict = env_dict["Config"]
        match env_dict["type"]:
            case "Gym":
                match export:
                    case None:
                        render_mode = None
                    case "render_only":
                        render_mode = "human"
                    case _:
                        render_mode = "rgb_array"
                config_dict["render_mode"] = render_mode
                if "seed" not in config_dict.keys():
                    config_dict["seed"] = seed
                return GymRenderableEnv.from_gym_make(**config_dict)
            case "ML-Agents":
                raise NotImplementedError("ML-Agents environment is not implemented yet")
            case _:
                raise AINEFactoryError("invalid environment type")
            
    @staticmethod
    def from_yaml(file_path: str) -> "AINEInferenceFactory":
        with open(file_path) as f:
            config = yaml.load(f, yaml.FullLoader)
        return AINEInferenceFactory(config)