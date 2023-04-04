from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import gym
import gym.spaces
import gym.vector
import numpy as np
import torch
from mlagents_envs.environment import (ActionTuple, BaseEnv, SideChannel,
                                       UnityEnvironment)
from mlagents_envs.registry import default_registry

from aine_drl.exp import Action, Observation


class ObservationSpace(tuple[int, ...]):
    """
    Observation space (shape tuple) of the environment.
    
    * vector space: `(num_features,)`
    * image space: `(height, width, num_channels)`
    """
    pass

@dataclass(frozen=True)
class ActionSpace:
    """
    Action space of the environment.
    
    * discrete action space: each element of the tuple is the number of actions in each action branch
    * continuous action space: the number of continuous actions
    """
    discrete: tuple[int, ...]
    continuous: int
    
    def sample(self, batch_size: int = 1) -> Action:
        discrete_actions = torch.cat(tuple(
            torch.randint(num_discrete_actions, (batch_size, 1)) for num_discrete_actions in self.discrete
        ), dim=-1) if len(self.discrete) > 0 else None
        continuous_actions = torch.randn(batch_size, self.continuous) if self.continuous > 0 else None
        return Action(discrete_actions, continuous_actions)

class Env(ABC):
    """AINE-DRL compatible reinforcement learning environment interface."""
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
            reward (Tensor): reward `(num_envs, 1)`
            terminated (Tensor): whether the episode is terminated `(num_envs, 1)`
            real_final_next_obs (Observation | None): "real" final next observation `(num_terminated_envs, *obs_shape)` of the episode. 
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
    def obs_spaces(self) -> tuple[ObservationSpace, ...]:
        """Returns the shape of the observation space."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        """Returns action space of the environment."""
        raise NotImplementedError
    
class Renderable(ABC):
    @abstractmethod
    def render(self) -> np.ndarray:
        raise NotImplementedError

class GymEnv(Env):
    """AINE-DRL compatiable OpenAI Gym environment."""
    def __init__(
        self, 
        env: gym.Env | gym.vector.VectorEnv,
        truncate_episode: bool = True,
        seed: int | list[int] | None = None
    ) -> None:
        if isinstance(env, gym.vector.VectorEnv):
            self._env = env
        else:
            self._env: gym.vector.VectorEnv = gym.vector.SyncVectorEnv(iter([lambda: env]))
        
        self._truncate_episode = truncate_episode
        self._seed = seed
        
        self._num_envs = self._env.num_envs
        self._action_converter = self._gym_action_converter()
        self._action_space = self._gym_action_space()
        
    def reset(self) -> Observation:
        return self._wrap_obs(self._env.reset(seed=self._seed)[0])
        
    def step(self, action: Action) -> tuple[Observation, torch.Tensor, torch.Tensor, Observation | None]:
        converted_action = self._action_converter(action)
        next_obs, reward, terminated, truncated, info = self._env.step(converted_action) # type: ignore
        real_final_next_obs = None
        if "final_observation" in info.keys():
            final_obs_mask = info["_final_observation"]
            
            if type(next_obs) == np.ndarray:
                real_final_next_obs = np.stack(info["final_observation"][final_obs_mask], axis=0)
            else:
                real_final_next_obs = []
                for i in range(len(next_obs)):
                    temp = np.stack(info["final_observation"][final_obs_mask][i], axis=0)
                    real_final_next_obs.append(temp)
        if self._truncate_episode:
            terminated |= truncated
        return (
            self._wrap_obs(next_obs),
            torch.from_numpy(reward).unsqueeze(dim=-1),
            torch.from_numpy(terminated).unsqueeze(dim=-1),
            self._wrap_obs(real_final_next_obs) if real_final_next_obs is not None else None
        )
        
    def close(self):
        self._env.close()
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def obs_spaces(self) -> tuple[ObservationSpace, ...]:
        if type(self._env.single_observation_space) == gym.spaces.Tuple:
            obs_shapes = []
            for obs_space in self._env.single_observation_space.spaces:
                if obs_space.shape is None:
                    raise ValueError("Observation space must be a tuple of Box or Discrete.")
                obs_shapes.append(ObservationSpace(obs_space.shape))
            return tuple(obs_shapes)
        
        obs_shape = self._env.single_observation_space.shape
        if obs_shape is None:
            raise ValueError("Observation space must be a tuple of Box or Discrete.")
        return (ObservationSpace(obs_shape),)
    
    @property
    def action_space(self) -> ActionSpace:
        return self._action_space
    
    def _gym_action_space(self) -> ActionSpace:
        num_discrete_actions = tuple()
        num_continuous_actions = 0
        action_space = self._env.single_action_space
        if isinstance(action_space, gym.spaces.Discrete):
            num_discrete_actions = (action_space.n,)
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            num_discrete_actions = tuple(action_space.nvec)
        elif isinstance(action_space, gym.spaces.Box):
            num_continuous_actions = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Tuple):
            if isinstance(action_space[0], gym.spaces.Discrete):
                num_discrete_actions = (action_space[0].n,) # type: ignore
            elif isinstance(action_space[0], gym.spaces.MultiDiscrete):
                num_discrete_actions = tuple(action_space[0].nvec.tolist()) # type: ignore
            else:
                raise RuntimeError(f"unsupported action space: {action_space[0]}")
            
            if isinstance(action_space[1], gym.spaces.Box):
                num_continuous_actions = action_space[1].shape[0] # type: ignore
            else:
                raise RuntimeError(f"unsupported action space: {action_space[1]}")
        else:
            raise NotImplementedError(f"{action_space} action space is not supported yet.")
        return ActionSpace(num_discrete_actions, num_continuous_actions)
    
    def _wrap_obs(self, obs) -> Observation:
        if type(obs) == np.ndarray:
            return Observation.from_tensor(torch.from_numpy(obs))
        else:
            return Observation(tuple(torch.from_numpy(o) for o in obs))
    
    def _gym_action_converter(self) -> Callable[[Action], Any]:
        action_space = self._env.action_space
        if isinstance(action_space, gym.spaces.Discrete) or isinstance(action_space, gym.spaces.MultiDiscrete):
            action_space_shape: tuple[int, ...] = self._env.action_space.shape # type: ignore
            return lambda a: a.discrete_action.reshape(action_space_shape).detach().cpu().numpy()
        elif isinstance(action_space, gym.spaces.Box):
            action_space_shape: tuple[int, ...] = self._env.action_space.shape # type: ignore
            return lambda a: a.continuous_action.reshape(action_space_shape).detach().cpu().numpy()
        elif isinstance(action_space, gym.spaces.Tuple):
            discrete_action_shape: tuple[int, ...] = self._env.action_space[0].shape # type: ignore
            continuous_action_shape = self._env.action_space[1].shape # type: ignore
            return lambda a: (
                a.discrete_action.reshape(discrete_action_shape).detach().cpu().numpy(), 
                a.continuous_action.reshape(continuous_action_shape).detach().cpu().numpy()
            )
        else:
            raise NotImplementedError(f"{self._env.single_action_space} action space is not supported yet.")
            
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
    """AINE-DRL compatible gym environment with render method."""
    def __init__(
        self, 
        env: gym.Env, 
        truncate_episode: bool = True, 
        seed: int | list[int] | None = None
    ) -> None:
        if env.render_mode == "human":
            self._renderer = lambda e: None
        elif env.render_mode == "rgb_array":
            self._renderer = lambda e: e.render()
        elif env.render_mode == "rgb_array_list":
            self._renderer = lambda e: e.render()[0]
        else:
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

class MLAgentsGymWrapper(gym.Env):
    """ML-Agents Gym Wrapper."""
    TRACE_AGENT: int = 0
    
    def __init__(
        self,
        env: BaseEnv
    ) -> None:
        self._env = env
        
        self._env.reset()
        self._behavior_name = tuple(self._env.behavior_specs)[0]
        spec = self._env.behavior_specs[self._behavior_name]
        
        self.observation_space = gym.spaces.Tuple(
            gym.spaces.Box(
                float("-inf"), 
                float("inf"),
                shape=obs_spec.shape
            ) for obs_spec in spec.observation_specs
        )
        
        self.action_space = gym.spaces.Tuple((
            gym.spaces.MultiDiscrete(list(
                spec.action_spec.discrete_branches
            )),
            gym.spaces.Box(
                float("-inf"),
                float("inf"),
                shape=(spec.action_spec.continuous_size,)
            )
        ))
        
    def reset(
        self, 
        *, 
        seed: int | None = None, 
        options: dict | None = None
    ) -> tuple[tuple[np.ndarray, ...], dict]:
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self._behavior_name)
        return tuple(decision_steps[self.TRACE_AGENT].obs), {}
    
    def step(
        self, 
        action: tuple[np.ndarray, np.ndarray]
    ) -> tuple[tuple[np.ndarray, ...], float, bool, bool, dict]:
        # take action
        action_tuple = ActionTuple(
            discrete=action[0][np.newaxis, ...],
            continuous=action[1][np.newaxis, ...]
        )
        self._env.set_action_for_agent(self._behavior_name, self.TRACE_AGENT, action_tuple)
        # until the traced agent is requested to take an action or terminated
        while True:
            self._env.step()
            # observe next observation and reward
            decision_steps, terminal_steps = self._env.get_steps(self._behavior_name)
            
            if self.TRACE_AGENT in terminal_steps:
                terminal_step = terminal_steps[self.TRACE_AGENT]
                
                next_obs = tuple(terminal_step.obs)
                reward = terminal_step.reward
                terminated = True
                truncated = True
                return next_obs, reward, terminated, truncated, {}
            
            if self.TRACE_AGENT in decision_steps:
                decision_step = decision_steps[self.TRACE_AGENT]
                
                next_obs = tuple(decision_step.obs)
                reward = decision_step.reward
                terminated = False
                truncated = False
                return next_obs, reward, terminated, truncated, {}
            
    def close(self):
        self._env.close()

class MLAgentsEnv(GymEnv):
    """AINE-DRL compatible ML-Agents environment."""
    def __init__(
        self, 
        env_factories: Iterable[Callable[[], BaseEnv]], 
        seed: int | list[int] | None = None
    ) -> None:
        gym_wrapper_factories = []
        for env_factory in env_factories:
            def gym_wrapper_factory(env_factory=env_factory):
                return MLAgentsGymWrapper(env_factory())
            gym_wrapper_factories.append(gym_wrapper_factory)
        env = gym.vector.AsyncVectorEnv(gym_wrapper_factories)
        super().__init__(env, True, seed)
        
    @staticmethod
    def from_registry(
        id: str, 
        num_envs: int = 1, 
        worker_id_start: int = 0,
        **kwargs
    ) -> "MLAgentsEnv":
        if "worker_id" in kwargs:
            raise ValueError("worker_id is not allowed to be specified in kwargs.")
        
        env_factories = []
        for worker_id in range(worker_id_start, worker_id_start + num_envs):
            def env_factory(worker_id=worker_id):
                warnings.filterwarnings("ignore")
                env = default_registry[id].make(worker_id=worker_id, **kwargs)
                warnings.filterwarnings("default")
                return env
            env_factories.append(env_factory)
        
        return MLAgentsEnv(env_factories)
    
    @staticmethod
    def from_unity_env(
        file_name: str | None = None,
        num_envs: int = 1,
        worker_id_start: int = 0,
        base_port: int | None = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 60,
        additional_args: list[str] | None = None,
        side_channels: list[SideChannel] | None = None,
        log_folder: str | None = None,
        num_areas: int = 1,
    ) -> "MLAgentsEnv":
        env_factories = []
        for worker_id in range(worker_id_start, worker_id_start + num_envs):
            def env_factory(worker_id=worker_id):
                warnings.filterwarnings("ignore")
                env = UnityEnvironment(
                    file_name,
                    worker_id,
                    base_port,
                    seed,
                    no_graphics,
                    timeout_wait,
                    additional_args,
                    side_channels,
                    log_folder,
                    num_areas
                )
                warnings.filterwarnings("default")
                return env
            env_factories.append(env_factory)
        return MLAgentsEnv(env_factories)
