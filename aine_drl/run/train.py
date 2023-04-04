from __future__ import annotations
import time
from dataclasses import dataclass

import torch

import aine_drl.util as util
from aine_drl.agent.agent import Agent, BehaviorScope, BehaviorType
from aine_drl.exp import Experience
from aine_drl.util.logger import TextInfoBox, logger

from aine_drl.env import Env
from aine_drl.run.error import AgentLoadError


@dataclass(frozen=True)
class TrainConfig:
    time_steps: int
    summary_freq: int
    agent_save_freq: int
    
    def __init__(
        self,
        time_steps: int,
        summary_freq: int | None = None,
        agent_save_freq: int | None = None,
    ) -> None:
        summary_freq = time_steps // 20 if summary_freq is None else summary_freq
        agent_save_freq = summary_freq * 10 if agent_save_freq is None else agent_save_freq
        
        object.__setattr__(self, "time_steps", time_steps)
        object.__setattr__(self, "summary_freq", summary_freq)
        object.__setattr__(self, "agent_save_freq", agent_save_freq)

class Train:
    def __init__(
        self,
        id: str,
        config: TrainConfig,
        env: Env,
        agent: Agent
    ) -> None:
        self._id = id
        self._config = config
        self._env = env
        self._agent = agent
        
        self._dtype = torch.float32
        self._device = self._agent.device
        self._trace_env = 0
        
        self._time_steps = 0
        self._episodes = 0
        self._episode_len = 0
        self._real_start_time = time.time()
        self._real_time = 0.0
        
        self._cumulative_reward_mean = util.IncrementalMean()
        self._episode_len_mean = util.IncrementalMean()
        
        self._enabled = True
        
    def train(self) -> "Train":
        if not self._enabled:
            raise RuntimeError("Train is already closed.")
        
        with BehaviorScope(self._agent, BehaviorType.TRAIN):
            if not logger.enabled():
                logger.enable(self._id, enable_log_file=False)
                
            self._load_train()
            
            if self._time_steps >= self._config.time_steps:  
                logger.print(f"Training is already finished.")
                return self
            
            logger.disable()
            logger.enable(self._id)
            
            self._print_train_info()            
            
            try:
                obs = self._env.reset().transform(self._agent_tensor)
                cumulative_reward = 0.0
                last_agent_save_t = 0
                for _ in range(self._time_steps, self._config.time_steps):
                    # take action and observe
                    action = self._agent.select_action(obs)
                    next_obs, reward, terminated, real_final_next_obs = self._env.step(action)
                    
                    # update the agent
                    next_obs = next_obs.transform(self._agent_tensor)
                    real_next_obs = next_obs.clone()
                    
                    if real_final_next_obs is not None:
                        real_next_obs[terminated.squeeze(dim=-1)] = real_final_next_obs.transform(self._agent_tensor)
                    
                    exp = Experience(
                        obs,
                        action,
                        real_next_obs,
                        self._agent_tensor(reward),
                        self._agent_tensor(terminated),
                    )
                    self._agent.update(exp)
                    
                    # take next step
                    obs = next_obs
                    cumulative_reward += reward[self._trace_env].item()
                    self._tick_time_steps()
                    
                    if terminated[self._trace_env].item():
                        self._cumulative_reward_mean.update(cumulative_reward)
                        self._episode_len_mean.update(self._episode_len)
                        
                        cumulative_reward = 0.0
                        self._tick_episode()
                    
                    # summary
                    if self._time_steps % self._config.summary_freq == 0:
                        self._summary_train()
                        
                    # save the agent
                    if self._time_steps % self._config.agent_save_freq == 0:
                        self._save_train()
                        last_agent_save_t = self._time_steps
                logger.print(f"Training is finished.")
                if self._time_steps > last_agent_save_t:
                    self._save_train()
                
            except KeyboardInterrupt:
                logger.print(f"Training interrupted at the time step {self._time_steps}.")
                self._save_train()
                    
        return self
    
    def close(self):
        self._enabled = False
        self._env.close()
        if logger.enabled():
            logger.disable()
    
    def _tick_time_steps(self):
        self._episode_len += 1
        self._time_steps += 1
        self._real_time = time.time() - self._real_start_time
    
    def _tick_episode(self):
        self._episode_len = 0
        self._episodes += 1
        
    def _agent_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self._device, dtype=self._dtype)
    
    def _print_train_info(self):
        text_info_box = TextInfoBox() \
            .add_text(f"AINE-DRL Training Start!") \
            .add_line(marker="=") \
            .add_text(f"ID: {self._id}") \
            .add_text(f"Output Path: {logger.log_dir()}") \
            .add_line() \
            .add_text(f"Training INFO:") \
            .add_text(f"    number of environments: {self._env.num_envs}") \
            .add_text(f"    total time steps: {self._config.time_steps}") \
            .add_text(f"    summary frequency: {self._config.summary_freq}") \
            .add_text(f"    agent save frequency: {self._config.agent_save_freq}") \
            .add_line() \
            .add_text(f"{self._agent.name} Agent:") \
            
        agent_config_dict = self._agent.config_dict
        agent_config_dict["device"] = self._device
            
        for key, value in self._agent.config_dict.items():
            text_info_box.add_text(f"    {key}: {value}")
            
        logger.print(text_info_box.make(), prefix="")
        logger.print("", prefix="")
        
    def _summary_train(self):
        if self._cumulative_reward_mean.count == 0:
            reward_info = "episode has not terminated yet"
        else:
            reward_info = f"cumulated reward: {self._cumulative_reward_mean.mean:.2f}"
            logger.log("Environment/Cumulative Reward", self._cumulative_reward_mean.mean, self._time_steps)
            logger.log("Environment/Episode Length", self._episode_len_mean.mean, self._time_steps)
            self._cumulative_reward_mean.reset()
            self._episode_len_mean.reset()
        logger.print(f"training time: {self._real_time:.2f}, time steps: {self._time_steps}, {reward_info}")
        
        for key, (value, t) in self._agent.log_data.items():
            logger.log(key, value, t)
            
    def _save_train(self):
        train_dict = dict(
            time_steps=self._time_steps,
            episodes=self._episodes,
            episode_len=self._episode_len,
        )
        state_dict = dict(
            train=train_dict,
            agent=self._agent.state_dict,
        )
        logger.save_agent(state_dict, self._time_steps)
        logger.print(f"Agent is successfully saved: {logger.agent_save_path()}")
    
    def _load_train(self):
        try:
            state_dict = logger.load_agent()
        except FileNotFoundError:
            return
        
        try:
            train_dict = state_dict["train"]
            self._time_steps = train_dict["time_steps"]
            self._episodes = train_dict["episodes"]
            self._episode_len = train_dict["episode_len"]
            self._agent.load_state_dict(state_dict["agent"])
        except:
            raise AgentLoadError("the loaded agent is not compatible with the current agent")