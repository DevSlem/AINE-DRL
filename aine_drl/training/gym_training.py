from typing import List, Union, Optional, NamedTuple

import gym.spaces as gym_space
import gym
import gym.vector
from gym import Env
from gym.vector import VectorEnv

from aine_drl.agent.agent import Agent, BehaviorType
from aine_drl.experience import Experience
from aine_drl.util import logger, ConfigManager
import aine_drl.training.gym_action_communicator as gac

import numpy as np

class GymTrainingConfig(NamedTuple):
    """
    dd

    Args:
        total_global_time_steps (int): total global time steps to train
        summary_freq (int): summary frequency
        agent_save_freq (int | None, optional): agent save frequency. Defaults to `summary_freq` x 10
        inference_freq (int | None, optional): inference frequency. Defaults to no inference.
        inference_render (bool, optional): whether render the environment when inference mode. Defaults to no rendering.
        auto_retrain (bool, optional): enable to retrain the agent. if you want to do, you must set custom `env_id`. Defaults to False.
        seed (int | List[int] | None, optional): gym environment random seed. Defaults to None.
    """
    total_global_time_steps: int
    summary_freq: int
    agent_save_freq: Optional[int] = None
    inference_freq: Optional[int] = None
    inference_render: bool = False
    auto_retrain: bool = False
    seed: Union[int, List[int], None] = None

class GymTraining:
    """
    Gym training class.

    Args:
        training_config (GymTrainingConfig): gym training configuration
        gym_env (Env | VectorEnv): gym environment
        gym_action_communicator (GymActionCommunicator | None, optional): action communicator between AINE-DRL and Gym. Defaults to auto set.
        env_id (str | None, optional): custom environment id. Defaults to `gym_env` id.
    """
    def __init__(self,
                 training_config: GymTrainingConfig,
                 gym_env: Union[Env, VectorEnv],
                 gym_action_communicator: Optional[gac.GymActionCommunicator] = None,
                 env_id: Optional[str] = None) -> None:
        
        assert training_config.total_global_time_steps >= 1
        assert training_config.summary_freq >= 1
        assert training_config.agent_save_freq is None or training_config.agent_save_freq >= 1
        assert training_config.inference_freq is None or training_config.inference_freq >= 1
        
        
        if env_id is None and training_config.auto_retrain == True:
            raise ValueError("You must set auto_retrain to False when env_id is None.")
        
        self.config = training_config
        self.gym_env = gym_env
                
        if isinstance(gym_env, VectorEnv):
            self.num_envs = gym_env.num_envs
            self.is_vector_env = True
        elif isinstance(gym_env, Env):
            self.num_envs = 1
            self.is_vector_env = False
        else:
            raise TypeError(f"You've instantiated GymTraining with {type(self.gym_env)} which isn't gym environment. Reinstantiate it.")
        
        assert gym_env.new_step_api == True  # type: ignore
        
        if gym_action_communicator is None:
            self.gym_action_communicator = gac.GymActionCommunicator.make(self.gym_env)
        else:
            self.gym_action_communicator = gym_action_communicator
        
        if self.config.agent_save_freq is None:
            self.config = self.config._replace(agent_save_freq=self.config.summary_freq * 10)
        
        self._agent_loaded = False
        self._logger_started = False
        self.env_id: str = ""
        self._set_env_id(env_id)
        
        self.dtype = np.float32
        
    @staticmethod
    def make(env_config: dict,
             env_id: Optional[str] = None,
             gym_env: Union[Env, VectorEnv, None] = None,
             gym_action_communicator: Optional[gac.GymActionCommunicator] = None) -> "GymTraining":
        """
        ## Summary
        
        Helps to make `GymTraining` instance.

        Args:
            env_config (dict): environment configuration which includes `num_envs`, `Gym`
            env_id (str | None, optional): custom environment id. Defaults to `gym_env` id.
            gym_env (Env | VectorEnv | None, optional): gym environment. Defaults to making it from the config.
            gym_action_communicator (GymActionCommunicator | None, optional): action communicator between AINE-DRL and Gym. Defaults to auto set.

        Returns:
            GymTraining: `GymTraining` instance
            
        ## Example
        
        `env_config` dictionary format::
        
            {'num_envs': 3,
             'Gym': {'env': {'id': 'CartPole-v1'},
              'training': {'seed': 0,
               'auto_retrain': True,
               'total_global_time_steps': 200000,
               'summary_freq': 1000,
               'agent_save_freq': None,
               'inference_freq': 10000,
               'inference_render': True}}
        
        `env_config` YAML format::
        
            num_envs: 3
            Gym:
              env:
                id: "CartPole-v1"
              training:
                seed: 0
                auto_retrain: false
                total_global_time_steps: 200000
                summary_freq: 1000
                agent_save_freq: null
                inference_freq: 10000
                inference_render: true
        """
        
        num_envs = env_config["num_envs"]
        gym_config = env_config["Gym"]
        gym_env_config = None
        if gym_env is None:
            gym_env_config = gym_config["env"]
            if num_envs > 1:
                gym_env = gym.vector.make(num_envs=num_envs, new_step_api=True, **gym_env_config)
            else:
                gym_env = gym.make(new_step_api=True, **gym_env_config)
                            
        training_config = GymTrainingConfig(**gym_config["training"])
        
        gym_training = GymTraining(
            training_config,
            gym_env,
            gym_action_communicator,
            env_id
        )
        
        if gym_env_config is not None and training_config.inference_freq is not None:
            render_mode = "human" if training_config.inference_render else None
            inference_gym_env = gym.make(new_step_api=True, render_mode=render_mode, **gym_env_config)
            inference_gym_action_communicator = gac.GymActionCommunicator.make(inference_gym_env)
            gym_training.set_inference_gym_env(inference_gym_env, inference_gym_action_communicator)
        
        return gym_training
        
        
    def train(self, agent: Agent, total_global_time_steps: Optional[int] = None):
        """
        Run training.

        Args:
            agent (Agent): DRL agent to train
            total_global_time_steps (int | None, optional): training total time steps. Defaults to internal setting.
        """
        if total_global_time_steps is None:
            if self.config.total_global_time_steps is None:
                raise ValueError("You must set either Agent.total_global_time_steps or total_global_time_steps parameter.")
            total_global_time_steps = self.config.total_global_time_steps
        
        if agent.clock.global_time_step >= total_global_time_steps:
            logger.print(f"Since {self.env_id} agent already reached to the total time steps, you can't train the agent.")
            return
        
        try:
            logger.start(self.env_id)
            self._logger_started = True
            if self.config.auto_retrain:
                self._try_load_agent(agent)
            
            self._train(agent, total_global_time_steps)
        except KeyboardInterrupt:
            logger.print(f"Training interrupted at the time step {agent.clock.global_time_step}.")
        finally:
            self._save_agent(agent)
            logger.end()
            self._logger_started = False
            
    def set_inference_gym_env(self, gym_env: Env, gym_action_communicator: gac.GymActionCommunicator):
        """Set manually the inference gym environment."""
        self.inference_gym_env = gym_env
        self.inference_gym_action_communicator = gym_action_communicator
            
    def inference(self, agent: Agent, num_episodes: int = 1):
        """
        Inference the environment.

        Args:
            num_episodes (int, optional): the number of inference episodes. Defaults to 1.
        """
        assert self.inference_gym_env is not None and self.inference_gym_action_communicator is not None, "You must call GymTraining.set_inference_gym_env() method when you want to inference."
        
        if not self._logger_started:
            logger.set_log_dir(self.env_id)
        self._try_load_agent(agent)
        
        seed: Optional[int] = self.config.seed if type(self.config.seed) is not list else self.config.seed[0]  # type: ignore
        
        agent.behavior_type = BehaviorType.INFERENCE
        for _ in range(num_episodes):
            # (num_envs, *obs_shape) = (1, *obs_shape)
            obs = self.inference_gym_env.reset(seed=seed).astype(self.dtype)[np.newaxis, ...]
            terminated = False
            cumulative_reward = 0.0
            while not terminated:
                action = agent.select_action(obs)
                next_obs, reward, teraminted, truncated, _ = self.inference_gym_env.step(self.inference_gym_action_communicator.to_gym_action(action))
                terminated = teraminted | truncated
                obs = next_obs.astype(self.dtype)[np.newaxis, ...]
                cumulative_reward += reward
            logger.print(f"inference mode - global time step {agent.clock.global_time_step}, cumulative reward: {cumulative_reward}")
            
        agent.behavior_type = BehaviorType.TRAIN
        
    def close(self):
        self.gym_env.close()
        if self.inference_gym_env is not None:
            self.inference_gym_env.close()
            
    @property
    def observation_space(self) -> gym_space.Space:
        return self.gym_env.single_observation_space if self.is_vector_env else self.gym_env.observation_space  # type: ignore
    
    @property
    def action_space(self) -> gym_space.Space:
        return self.gym_env.single_action_space if self.is_vector_env else self.gym_env.action_space  # type: ignore
        
    def _train(self, agent: Agent, total_global_time_steps: int):
        gym_env = self.gym_env
        obs = gym_env.reset(seed=self.config.seed).astype(self.dtype)
        if not self.is_vector_env:
            # (num_envs, *obs_shape) = (1, *obs_shape)
            obs = obs[np.newaxis, ...]
        
        logger.print("Training start!")
        for _ in range(agent.clock.global_time_step, total_global_time_steps, self.num_envs):
            # take action and observe
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.gym_env.step(self.gym_action_communicator.to_gym_action(action))
            terminated = terminated | truncated
            
            # update the agent
            exp = self._make_experience(obs, action, next_obs, reward, terminated)
            agent.update(exp)
            
            # update current observation
            if not self.is_vector_env and terminated:
                obs = gym_env.reset(seed=self.config.seed).astype(self.dtype)[np.newaxis, ...]
            else:
                obs = exp.next_obs
                
            # summuary check
            if agent.clock.check_global_time_step_freq(self.config.summary_freq):
                self._summary(agent)
                
            if agent.clock.check_global_time_step_freq(self.config.agent_save_freq):  # type: ignore
                self._save_agent(agent)
                
            # inference check
            if self.config.inference_freq is not None and agent.clock.check_global_time_step_freq(self.config.inference_freq):
                self.inference(agent)
        
        logger.print("Training has been completed.")
            
    def _set_env_id(self, env_id: Optional[str]):
        """ set environment id. """
        if env_id is None:
            env_id = self.gym_env.get_attr("spec")[0].id if self.is_vector_env else self.gym_env.spec.id
        self.env_id = env_id if self.config.auto_retrain else logger.numbering_env_id(env_id)
            
    def _save_agent(self, agent: Agent):
        try:
            logger.save_agent(agent.state_dict)
            logger.print(f"Saving the current agent is successfully completed: {logger.agent_save_dir()}")
        except FileNotFoundError:
            pass
            
    def _try_load_agent(self, agent: Agent):
        if not self._agent_loaded:
            try:
                ckpt = logger.load_agent()
                agent.load_state_dict(ckpt)
                self._agent_loaded = True
                logger.print(f"Loading the saved agent is successfully completed: {logger.agent_save_dir()}")
            except FileNotFoundError:
                pass
                
    def _summary(self, agent: Agent):
        log_data = agent.log_data
        global_time_step = agent.clock.global_time_step
        if "Environment/Cumulative Reward" in log_data.keys():
            logger.print(
                f"training time: {agent.clock.real_time:.1f}, global time step: {global_time_step}, cumulative reward: {log_data['Environment/Cumulative Reward'][0]:.1f}"
            )
        else:
            logger.print(f"training time: {agent.clock.real_time:.1f}, global time step: {global_time_step}, episode has not terminated yet.")
            
        for key, value in log_data.items():
            logger.log(key, value[0], value[1])
                
    def _make_experience(self, obs, action, next_obs, reward, terminated) -> Experience:
        if self.is_vector_env:
            exp = Experience(
                obs.astype(np.float32),
                action,
                next_obs.astype(np.float32),
                reward.astype(np.float32)[..., np.newaxis],
                terminated.astype(np.float32)[..., np.newaxis]
            )
        else:
            exp = Experience(
                obs.astype(np.float32),
                action,
                next_obs[np.newaxis, ...].astype(np.float32),
                np.array([[reward]], dtype=np.float32),
                np.array([[terminated]], dtype=np.float32)
            )
        return exp
