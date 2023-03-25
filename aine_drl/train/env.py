from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import gym
import gym.spaces
import gym.vector
import numpy as np
import torch

from aine_drl.exp import Action, Observation


@dataclass(frozen=True)
class ActionSpec:
    num_discrete_actions: tuple[int, ...]
    num_continuous_actions: int

class Env(ABC):
    @abstractmethod
    def reset(self) -> Observation:
        """
        Resets the environment to an initial state and returns the initial observation.

        Returns:
            obs (Observation): observation of the initial state `(num_envs, *obs_shape)`
        """
        raise NotImplementedError 
    
    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, torch.Tensor, torch.Tensor, Observation | None]:
        """
        Takes a step in the environment using an action.

        Args:
            action (Action): an action provided by the agent

        Returns:
            next_obs (Observation): next observation `(num_envs, *obs_shape)` which is automatically reset to the first observation of the next episode
            reward (Tensor): reward `(num_envs, 1en)`
            terminated (Tensor): whether the episode is terminated `(num_envs, 1)`
            real_final_next_obs (Observation): next observation `(num_envs, *obs_shape)` which includes the "real" final observation of the episode. 
            You can access only if any environment is terminated.
        """
        raise NotImplementedError
    
    @abstractmethod
    def close(self):
        """Close all environments and release resources."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Returns the number of environments."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def obs_shape(self) -> tuple[int, ...]:
        """Returns the shape of the observation space."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def action_spec(self) -> ActionSpec:
        raise NotImplementedError
    
class Renderable(ABC):
    @abstractmethod
    def render(self) -> np.ndarray:
        raise NotImplementedError

class GymEnv(Env):
    def __init__(
        self, 
        env: gym.Env | gym.vector.VectorEnv,
        truncate_episode: bool = True,
        seed: int | list[int] | None = None
    ) -> None:
        if isinstance(env, gym.vector.VectorEnv):
            self.env = env
        else:
            self.env: gym.vector.VectorEnv = gym.vector.SyncVectorEnv(iter([lambda: env]))
        
        self._truncate_episode = truncate_episode
        self._seed = seed
        
        self._num_envs = self.env.num_envs
        self._action_converter = self._gym_action_converter()
        self._action_spec = self._gym_action_spec()
        
    def reset(self) -> Observation:
        return self._wrap_obs(self.env.reset(seed=self._seed)[0])
        
    def step(self, action: Action) -> tuple[Observation, torch.Tensor, torch.Tensor, Observation | None]:
        converted_action = self._action_converter(action)
        next_obs, reward, terminated, truncated, info = self.env.step(converted_action) # type: ignore
        real_final_next_obs = None
        if "final_observation" in info.keys():
            real_final_next_obs = next_obs.copy()
            final_obs_mask = info["_final_observation"]
            real_final_next_obs[final_obs_mask] = np.stack(info["final_observation"][final_obs_mask], axis=0)
        if self._truncate_episode:
            terminated |= truncated
        return (
            self._wrap_obs(next_obs),
            torch.from_numpy(reward).unsqueeze(dim=-1),
            torch.from_numpy(terminated).unsqueeze(dim=-1),
            self._wrap_obs(real_final_next_obs) if real_final_next_obs is not None else None
        )
        
    def close(self):
        self.env.close()
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self.env.single_observation_space.shape # type: ignore
    
    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec
    
    def _gym_action_spec(self) -> ActionSpec:
        num_discrete_actions = tuple()
        num_continuous_actions = 0
        match type(self.env.single_action_space):
            case gym.spaces.Discrete:
                num_discrete_actions = (self.env.single_action_space.n,) # type: ignore
            case gym.spaces.MultiDiscrete:
                num_discrete_actions = tuple(self.env.single_action_space.nvec) # type: ignore
            case gym.spaces.Box:
                num_continuous_actions = self.env.single_action_space.shape[0] # type: ignore
            case _:
                raise NotImplementedError(f"{self.env.single_action_space} action space is not supported yet.")
        return ActionSpec(num_discrete_actions, num_continuous_actions)
    
    def _wrap_obs(self, obs) -> Observation:
        return Observation.from_tensor(torch.from_numpy(obs))
    
    def _gym_action_converter(self) -> Callable[[Action], Any]:
        action_space_shape: tuple[int, ...] = self.env.action_space.shape # type: ignore
        match type(self.env.action_space):
            case gym.spaces.Discrete | gym.spaces.MultiDiscrete:
                return lambda a: a.discrete_action.reshape(action_space_shape).detach().cpu().numpy()
            case gym.spaces.Box:
                return lambda a: a.continuous_action.reshape(action_space_shape).detach().cpu().numpy()
            case _:
                raise NotImplementedError(f"{self.env.single_action_space} action space is not supported yet.")
            
    @staticmethod
    def from_gym_make(
        id: str,
        num_envs: int = 1,
        truncate_episode: bool = True,
        seed: int | list[int] | None = None,
        asynchronous: bool = True,
        wrappers: Any = None, # type: ignore
        disable_env_checker: bool | None = None,
        **kwargs
    ):
        env = gym.vector.make(
            id, 
            num_envs, 
            asynchronous, 
            wrappers, 
            disable_env_checker,
            **kwargs
        )
        return GymEnv(env, truncate_episode, seed)

class GymRenderableEnv(GymEnv, Renderable):
    def __init__(
        self, 
        env: gym.Env, 
        truncate_episode: bool = True, 
        seed: int | list[int] | None = None
    ) -> None:
        match env.render_mode:
            case "human":
                self._renderer = lambda e: None
            case "rgb_array":
                self._renderer = lambda e: e.render()
            case "rgb_array_list":
                self._renderer = lambda e: e.render()[0]
            case _:
                raise NotImplementedError(f"{env.render_mode} render mode is not supported yet.")
                
        self._renderable_env = env
        super().__init__(env, truncate_episode, seed)
    
    def render(self) -> np.ndarray | None:
        return self._renderer(self._renderable_env)
    
    @staticmethod
    def from_gym_make(
        id: str, 
        truncated_episode: bool = True,
        seed: int | list[int] | None = None,
        **kwargs
    ):
        env = gym.make(id, **kwargs)
        return GymRenderableEnv(env, truncated_episode, seed)