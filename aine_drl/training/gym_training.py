from argparse import ArgumentTypeError
from typing import List, Union, Optional

import gym.spaces as gym_space
import gym
from gym import Env
from gym.vector import VectorEnv

from aine_drl.agent.agent import Agent, BehaviorType
from aine_drl.experience import Experience
from aine_drl.util import logger, ConfigManager
import aine_drl.training.gym_action_communicator as gac

import numpy as np

class GymTraining:
    """
    Gym training class.

    Args:
        gym_env (Env | VectorEnv): gym environment
        gym_action_communicator (GymActionCommunicator | None, optional): action communicator between AINE-DRL and Gym. Defaults to auto set.
        env_id (str | None, optional): custom environment id. Defaults to `gym_env` id.
        seed (int | List[int] | None, optional): gym environment random seed. Defaults to None.
        auto_retrain (bool, optional): enable to retrain the agent. if you want to do, you must set custom `env_id`. Defaults to False.
        total_global_time_steps (int | None, optional): total global time steps to train. Defaults to None.
        summary_freq (int | None, optional): summary frequency. Defaults to not summary.
        agent_save_freq (int | None, optional): agent save frequency. Defaults to `summary_freq` x 10
        inference_freq (int | None, optional): inference frequency. Defaults to not inference.
        
    """
    def __init__(self,
                 gym_env: Union[Env, VectorEnv],
                 gym_action_communicator: Optional[gac.GymActionCommunicator] = None,
                 env_id: Optional[str] = None,
                 seed: Union[int, List[int], None] = None,
                 auto_retrain: bool = False,
                 total_global_time_steps: Optional[int] = None,
                 summary_freq: Optional[int] = None,
                 agent_save_freq: Optional[int] = None,
                 inference_freq: Optional[int] = None) -> None:
        
        if env_id is None and auto_retrain == True:
            raise ValueError("You must set auto_retrain to False when env_id is None.")
        
        self.gym_env = gym_env
        
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
            self.gym_action_communicator = gac.GymActionCommunicator.make(self.gym_env)
        else:
            self.gym_action_communicator = gym_action_communicator
        
        self.seed = seed
        self.auto_retrain = auto_retrain
        self.total_global_time_steps = total_global_time_steps
        self.summary_freq = summary_freq
        self.agent_save_freq = summary_freq * 10 if agent_save_freq is None else agent_save_freq
        self.inference_freq = inference_freq
        
        self._agent_loaded = False
        self.inference_gym_env = None
        self.inference_gym_action_communicator = None
        self.save_on = True
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
        if gym_env is None:
            gym_env_config = gym_config["env"]
            if num_envs > 1:
                gym_env = gym.vector.make(num_envs=num_envs, new_step_api=True, **gym_env_config)
            else:
                gym_env = gym.make(new_step_api=True, **gym_env_config)
                            
        training_config = gym_config["training"]
        
        seed = training_config.get("seed", None)
        auto_retrain = training_config.get("auto_retrain", False)
        total_global_time_steps = training_config.get("total_global_time_steps", None)
        summary_freq = training_config.get("summary_freq", None)
        agent_save_freq = training_config.get("agent_save_freq", None)
        inference_freq = training_config.get("inference_freq", None)
        inference_render = training_config.get("inference_render", True)
        
        gym_training = GymTraining(
            gym_env=gym_env,
            gym_action_communicator=gym_action_communicator,
            env_id=env_id,
            seed=seed,
            auto_retrain=auto_retrain,
            total_global_time_steps=total_global_time_steps,
            summary_freq=summary_freq,
            agent_save_freq=agent_save_freq,
            inference_freq=inference_freq
        )
        
        if inference_freq is not None:
            render_mode = "human" if inference_render else None
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
            if self.total_global_time_steps is None:
                raise ValueError("You must set either Agent.total_global_time_steps or total_global_time_steps parameter.")
            total_global_time_steps = self.total_global_time_steps
        
        try:
            logger.start(self.env_id)
            if self.auto_retrain:
                self._load_agent(agent)
            
            self._agent_loaded = True
            self._train(agent, total_global_time_steps)
        except KeyboardInterrupt:
            logger.print(f"Training interrupted at the time step {agent.clock.global_time_step}.")
        finally:
            self._save_agent(agent)
            logger.end()
            
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
        
        self._load_agent(agent)
        
        agent.behavior_type = BehaviorType.INFERENCE
        for _ in range(num_episodes):
            # (num_envs, *obs_shape) = (1, *obs_shape)
            obs = self.inference_gym_env.reset(seed=self.seed).astype(self.dtype)[np.newaxis, ...]
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
        
    def _train(self, agent: Agent, total_gloabl_time_steps: int):
        gym_env = self.gym_env
        obs = gym_env.reset(seed=self.seed).astype(self.dtype)
        if not self.is_vector_env:
            # (num_envs, *obs_shape) = (1, *obs_shape)
            obs = obs[np.newaxis, ...]
        
        if agent.clock.global_time_step >= total_gloabl_time_steps:
            logger.print(f"Since {self.env_id} agent already reached to the total time steps, you can't train the agent.")
            self.save_on = False
            return
        
        logger.print("Training start!")
        for _ in range(agent.clock.global_time_step, total_gloabl_time_steps, self.num_envs):
            # take action and observe
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.gym_env.step(self.gym_action_communicator.to_gym_action(action))
            terminated = terminated | truncated
            
            # update the agent
            exp = self._make_experience(obs, action, next_obs, reward, terminated)
            agent.update(exp)
            
            # update current observation
            if not self.is_vector_env and terminated:
                obs = gym_env.reset(seed=self.seed).astype(self.dtype)[np.newaxis, ...]
            else:
                obs = exp.next_obs
                
            # summuary check
            if self.summary_freq is not None and agent.clock.check_global_time_step_freq(self.summary_freq):
                self._summary(agent)
                
            if agent.clock.check_global_time_step_freq(self.agent_save_freq):
                self._save_agent(agent)
                
            # inference check
            if self.inference_freq is not None and agent.clock.check_global_time_step_freq(self.inference_freq):
                self.inference(agent)
        
        logger.print("Training has been completed.")
            
    def _set_env_id(self, env_id: Optional[str]):
        """ set environment id. """
        if env_id is None:
            gym_env_id = self.gym_env.get_attr("spec")[0].id if self.is_vector_env else self.gym_env.spec.id
            env_id = gym_env_id
        self.env_id = env_id if self.auto_retrain else logger.numbering_env_id(env_id)
            
    def _save_agent(self, agent: Agent):
        if self.save_on:
            try:
                logger.save_agent(agent.state_dict)
                logger.print(f"Saving the current agent is successfully completed: {logger.agent_save_dir()}")
            except FileNotFoundError:
                pass
            
    def _load_agent(self, agent: Agent):
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
                reward.astype(np.float32),
                terminated.astype(np.float32)
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
