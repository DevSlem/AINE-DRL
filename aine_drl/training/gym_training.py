from argparse import ArgumentTypeError
from abc import ABC, abstractmethod
import gym
import gym.spaces as gym_space
from gym import Env
from gym.vector import VectorEnv
from typing import Any, List, Union, Optional
from aine_drl.agent.agent import Agent, BehaviorType
from aine_drl.experience import Action, Experience
import aine_drl.util as util
from aine_drl.util import logger
import torch
import numpy as np

class GymActionCommunicator(ABC):
    @abstractmethod
    def to_gym_action(self, action: Action) -> Any:
        raise NotImplementedError
    
class GymDiscreteActionCommunicator(GymActionCommunicator):
    def __init__(self, action_space: gym_space.Space) -> None:
        self.action_shape = action_space.shape
    
    def to_gym_action(self, action: Action) -> Any:
        return action.discrete_action.reshape(self.action_shape)

class GymTraining:
    """
    Gym training class.

    Args:
        agent (Agent): DRL agent to train
        gym_env (Env | VectorEnv): gym environment
        env_id (str | None, optional): custom environment id. Defaults to `gym_env` id.
        seed (int | List[int] | None, optional): gym environment random seed
        auto_retrain (bool, optional): enable to retrain the agent. if you want to do, you must set custom `env_id`. Defaults to False.
        render_freq (int | None, optional): inference rendering frequency. Defaults to never inference.
    """
    def __init__(self,
                 agent: Agent,
                 gym_env: Union[Env, VectorEnv],
                 env_id: Optional[str] = None,
                 seed: Union[int, List[int], None] = None,
                 auto_retrain: bool = False,
                 summary_freq: Optional[int] = None,
                 inference_freq: Optional[int] = None,
                 gym_action_communicator: Optional[GymActionCommunicator] = None) -> None:
        
        if env_id is None and auto_retrain == True:
            raise ValueError("You must set auto_retrain to False when env_id is None.")
        
        self.auto_retrain = auto_retrain
        self.gym_env = gym_env
        self.summary_freq = summary_freq
        self.inference_freq = inference_freq
        
        if isinstance(gym_env, VectorEnv):
            self.num_envs = gym_env.num_envs
            self.is_vector_env = True
        elif isinstance(gym_env, Env):
            self.num_envs = 1
            self.is_vector_env = False
        else:
            raise ArgumentTypeError(f"You've instantiated GymTraining with {type(self.gym_env)} which isn't gym environment. Reinstantiate it.")
        
        gym_env.new_step_api = True
        
        if gym_action_communicator is None:
            self.gym_action_communicator = self.make_gym_action_communicator(self.gym_env)
        else:
            self.gym_action_communicator = gym_action_communicator
        
        self.seed = seed
        self.agent = agent
        self._agent_loaded = False
        self.inference_gym_env = None
        self._set_env_id(env_id)
        
    def make_gym_action_communicator(self, gym_env: Union[Env, VectorEnv]) -> GymActionCommunicator:
        action_space_type = type(gym_env.action_space)
        if action_space_type is gym_space.Discrete or action_space_type is gym_space.MultiDiscrete:
            return GymDiscreteActionCommunicator(gym_env.action_space)
        else:
            raise ValueError("Doesn't implement yet for this action space.")
        
    def train(self, total_global_time_steps: int):
        """
        Run training.

        Args:
            total_gloabl_time_steps (int): training total time steps
        """
        try:
            # if self.auto_retrain:
            #     self._load_agent()
            logger.start(self.env_id)
            self._train(total_global_time_steps)
        except KeyboardInterrupt:
            logger.print(f"Training interrupted at the time step {self.agent.clock.global_time_step}.")
        finally:
            logger.end()
            # self._save_agent()
            
    def inference(self, num_episodes: int = 1):
        """
        Inference the environment.

        Args:
            num_episodes (int, optional): the number of inference episodes. Defaults to 1.
        """
        self.agent.behavior_type = BehaviorType.INFERENCE
        
        # self._load_agent()
        # for _ in range(num_episodes):
        #     obs = self.inference_gym_env.reset(seed=self.seed)
        #     obs = obs[np.newaxis, ...] # (num_envs, *obs_shape) = (1, *obs_shape)
        #     terminated = False
        #     cumulative_reward = 0.0
        #     while not terminated:
        #         action = self.agent.select_action(obs)
        #         next_obs, reward, teraminted, truncated, _ = self.__gym_render_env.step(self.gym_action_communicator.to_gym_action(action))
        #         # self.__gym_render_env.render()
        #         terminated = teraminted | truncated
        #         obs = next_obs
        #         cumulative_reward += reward
        #     logger.print(f"inference mode - time step {self.agent.clock.time_step}, cumulative reward: {cumulative_reward}")
            
        self.agent.behavior_type = BehaviorType.TRAIN
            
    def set_inference_gym_env(self, inference_gym_env: Env, is_render: bool = True):
        self.inference_gym_env = inference_gym_env
        self.is_render = is_render
        self.inference_gym_env.render_mode = "human" if is_render else None
            
    def _set_env_id(self, env_id: Optional[str]):
        """ set environment id. """
        if env_id is None:
            gym_env_id = self.gym_env.get_attr("spec")[0].id if self.is_vector_env else self.gym_env.spec.id
            env_id = gym_env_id
        self.env_id = env_id if self.auto_retrain else logger.numbering_env_id(env_id)
            
    def _save_agent(self):
        try:
            torch.save(self.agent.state_dict, logger.agent_save_dir())
        except FileNotFoundError:
            pass
            
    def _load_agent(self):
        if not self._agent_loaded:
            try:
                ckpt = torch.load(logger.agent_save_dir())
                self.agent.load_state_dict(ckpt)
                self._agent_loaded = True
            except FileNotFoundError:
                pass
    
    def _train(self, total_gloabl_time_steps: int):
        gym_env = self.gym_env
        obs = gym_env.reset(seed=self.seed)
        if self.agent.clock.global_time_step >= total_gloabl_time_steps:
            logger.print(f"Since {self.env_id} agent already reached to the total time steps, you can't train the agent.")
        for _ in range(self.agent.clock.global_time_step, total_gloabl_time_steps, self.num_envs):
            # take action and observe
            action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.gym_env.step(self.gym_action_communicator.to_gym_action(action))
            terminated = terminated | truncated
            
            # update the agent
            exp = self._make_experience(obs, action, next_obs, reward, terminated)
            self.agent.update(exp)
            
            # update current observation
            if (not self.is_vector_env) and terminated:
                obs = gym_env.reset(seed=self.seed)
            else:
                obs = next_obs
                
            # summuary check
            if self.summary_freq is not None and self.agent.clock.check_global_time_step_freq(self.summary_freq):
                self._summary()
                
            # inference check
            if self.inference_freq is not None and self.agent.clock.check_global_time_step_freq(self.inference_freq):
                self.inference()
                
    def _summary(self):
        log_data = self.agent.log_data
        global_time_step = self.agent.clock.global_time_step
        if "Environment/Cumulative Reward" in log_data.keys():
            logger.print(
                f"training time: {self.agent.clock.real_time:.1f}, global time step: {global_time_step}, cumulative reward: {log_data['Environment/Cumulative Reward'][0]:.1f}"
            )
        else:
            logger.print(f"training time: {self.agent.clock.real_time:.1f}, global time step: {global_time_step}, episode has not terminated yet.")
            
        for key, value in log_data.items():
            logger.log(key, value[0], value[1])
                
    def _make_experience(self, obs, action, next_obs, reward, terminated) -> Experience:
        if self.is_vector_env:
            exp = Experience(
                obs.astype(np.float32),
                action,
                next_obs.astype(np.float32),
                reward.astype(np.float32),
                terminated.astype(np.float32)
            )
        else:
            exp = Experience(
                obs[np.newaxis, ...].astype(np.float32),
                action,
                next_obs[np.newaxis, ...].astype(np.float32),
                np.array([[reward]], dtype=np.float32),
                np.array([[terminated]], dtype=np.float32)
            )
        return exp