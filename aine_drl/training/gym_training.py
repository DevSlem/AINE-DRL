from typing import List, Union, Optional, NamedTuple
import gym.spaces as gym_space
import gym
import gym.vector
from gym import Env
from gym.vector import VectorEnv
from aine_drl.agent.agent import Agent, BehaviorType
from aine_drl.experience import Experience
from aine_drl.util import logger
import aine_drl.training.gym_action_communicator as gac
import numpy as np
import torch

class GymTrainingConfig(NamedTuple):
    """
    dd

    Args:
        total_global_time_steps (int): total global time steps to train
        summary_freq (int): summary frequency
        agent_save_freq (int | None, optional): agent save frequency. Defaults to `summary_freq` x 10
        inference_freq (int | None, optional): inference frequency. Defaults to no inference.
        inference_render (bool, optional): whether render the environment when inference mode. Defaults to no rendering.
        generate_new_training_result (bool, optional): whether or not it generates new training result files. Defaults to False.
        seed (int | List[int] | None, optional): gym environment random seed. Defaults to None.
    """
    total_global_time_steps: int
    summary_freq: int
    agent_save_freq: Optional[int] = None
    inference_freq: Optional[int] = None
    inference_render: bool = False
    generate_new_training_result: bool = False
    seed: Union[int, List[int], None] = None

class GymTraining:
    """
    Gym training class.

    Args:
        training_env_id (str): training environment id
        training_config (GymTrainingConfig): gym training configuration
        gym_env (Env | VectorEnv): gym environment
        gym_action_communicator (GymActionCommunicator | None, optional): action communicator between AINE-DRL and Gym. Defaults to auto set.
    """
    def __init__(self,
                 training_env_id: str,
                 training_config: GymTrainingConfig,
                 gym_env: Union[Env, VectorEnv],
                 gym_action_communicator: Optional[gac.GymActionCommunicator] = None) -> None:
        
        assert training_config.total_global_time_steps >= 1
        assert training_config.summary_freq >= 1
        assert training_config.agent_save_freq is None or training_config.agent_save_freq >= 1
        assert training_config.inference_freq is None or training_config.inference_freq >= 1
        
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
        self.training_env_id = training_env_id if not self.config.generate_new_training_result else logger.numbering_env_id(training_env_id)
        
        self.inference_gym_env = None
        self.inference_gym_action_communicator = None
        
        self.dtype = np.float32
        
    @staticmethod
    def make(training_env_id: str,
             gym_config: dict,
             num_envs: Optional[int] = None,
             gym_env: Union[Env, VectorEnv, None] = None,
             gym_action_communicator: Optional[gac.GymActionCommunicator] = None) -> "GymTraining":
        """
        ## Summary
        
        Helps to make `GymTraining` instance.

        Args:
            training_env_id (str): training environment id
            gym_config (dict): gym environment configuration whose the main key `Gym` includes `env` or `training` 
            gym_env (Env | VectorEnv | None, optional): gym environment. Defaults to making it from the configuration.
            gym_action_communicator (GymActionCommunicator | None, optional): action communicator between AINE-DRL and Gym. Defaults to auto set.

        Returns:
            GymTraining: `GymTraining` instance
            
        ## Example
        
        `training_env_config` dictionary format::
        
            {'Gym': {'env': {'id': 'CartPole-v1'},
              'training': {'seed': 0,
               'auto_retrain': True,
               'total_global_time_steps': 200000,
               'summary_freq': 1000,
               'agent_save_freq': None,
               'inference_freq': 10000,
               'inference_render': True}}
        """
        
        gym_config = gym_config["Gym"]
        gym_env_config = None
        if gym_env is None:
            if num_envs is None or num_envs < 1:
                raise ValueError(f"If you want to make a gym environment using the configuration, you must set valid num_envs parameter, but you've set {num_envs}.")
            
            gym_env_config = gym_config["env"]
            if num_envs > 1:
                gym_env = gym.vector.make(num_envs=num_envs, new_step_api=True, **gym_env_config)
            else:
                gym_env = gym.make(new_step_api=True, **gym_env_config)
                            
        training_config = GymTrainingConfig(**gym_config["training"])
        
        gym_training = GymTraining(
            training_env_id,
            training_config,
            gym_env,
            gym_action_communicator
        )
        
        if gym_env_config is not None:
            gym_env_config["render_mode"] = "human" if training_config.inference_render else None
            inference_gym_env = gym.make(new_step_api=True, **gym_env_config)
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
            logger.print(f"Since {self.training_env_id} already reached to the total global time steps, you can't train the agent.")
            return
        
        try:
            logger.start(self.training_env_id)
            self._logger_started = True
            if not self.config.generate_new_training_result:
                self._try_load_agent(agent)
            self._agent_loaded = True
            
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
            
    def inference(self, agent: Agent, num_episodes: int = 1, agent_save_file_dir: Optional[str] = None):
        """
        Inference the environment.

        Args:
            agent (Agent): DRL agent to inference
            num_episodes (int, optional): the number of inference episodes. Defaults to 1.
            agent_save_file_dir (str, optional): set the agent save file directory when you want to load manually one. Defaults to auto loading.
        """
        try:            
            if agent_save_file_dir is None:
                logger.print(f"\'{self.training_env_id}\' inference start!")
            else:
                logger.print(f"\'{agent_save_file_dir}\' inference start!")
            self._inference(agent, num_episodes, agent_save_file_dir)
        except KeyboardInterrupt:
            logger.print(f"Inference interrupted.")
        
    def _inference(self, agent: Agent, num_episodes: int = 1, agent_save_file_dir: Optional[str] = None):
        assert self.inference_gym_env is not None and self.inference_gym_action_communicator is not None, "You must call GymTraining.set_inference_gym_env() method when you want to inference."
        
        if not self._logger_started:
            logger.set_log_dir(self.training_env_id)
        self._try_load_agent(agent, agent_save_file_dir=agent_save_file_dir)
        
        seed: Optional[int] = self.config.seed if type(self.config.seed) is not list else self.config.seed[0]  # type: ignore
        
        agent.behavior_type = BehaviorType.INFERENCE
        
        for episode in range(num_episodes):
            # (num_envs, *obs_shape) = (1, *obs_shape)
            obs = self.inference_gym_env.reset(seed=seed).astype(self.dtype)[np.newaxis, ...]
            terminated = False
            cumulative_reward = 0.0
            while not terminated:
                # take action and observe
                action = agent.select_action(obs)
                next_obs, reward, teraminted, truncated, _ = self.inference_gym_env.step(self.inference_gym_action_communicator.to_gym_action(action))
                terminated = teraminted | truncated
                
                # cumulate the reward
                cumulative_reward += reward
                
                # inference the agent
                exp = self._make_experience(obs, action, next_obs, reward, terminated, is_vector_env=False)
                agent.inference(exp)
                obs = exp.next_obs
            logger.print(f"inference mode - episode: {episode}, cumulative reward: {cumulative_reward}")
            
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
        logger.print(f"\'{self.training_env_id}\' training start!")
        gym_env = self.gym_env
        obs = gym_env.reset(seed=self.config.seed).astype(self.dtype)
        if not self.is_vector_env:
            # (num_envs, *obs_shape) = (1, *obs_shape)
            obs = obs[np.newaxis, ...]
        
        for _ in range(agent.clock.global_time_step, total_global_time_steps, self.num_envs):
            # take action and observe
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.gym_env.step(self.gym_action_communicator.to_gym_action(action))
            terminated = terminated | truncated
            
            # update the agent
            exp = self._make_experience(obs, action, next_obs, reward, terminated, self.is_vector_env)
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
                self._inference(agent)
        
        logger.print("Training has been completed.")
            
    def _save_agent(self, agent: Agent):
        try:
            logger.save_agent(agent.state_dict)
            logger.print(f"Saving the current agent is successfully completed: {logger.agent_save_dir()}")
        except FileNotFoundError:
            pass
            
    def _try_load_agent(self, agent: Agent, agent_save_file_dir: Optional[str] = None):
        if agent_save_file_dir is not None:
            try:
                ckpt = torch.load(agent_save_file_dir)
                agent.load_state_dict(ckpt)
            except FileNotFoundError as ex:
                raise FileNotFoundError(f"The agent save file directory is invalid: {agent_save_file_dir}") from ex
            return
        
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
                
    def _make_experience(self, obs, action, next_obs, reward, terminated, is_vector_env: bool) -> Experience:
        if is_vector_env:
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