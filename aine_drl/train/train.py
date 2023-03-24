from dataclasses import dataclass
from .env import Env
import torch
from aine_drl.exp import Experience
from aine_drl.agent.agent import Agent, BehaviorType
import time
from aine_drl.util.logger import logger, TextInfoBox
from aine_drl.util.util_methods import IncrementalAverage

class InferenceError(Exception):
    pass

@dataclass(frozen=True)
class TrainConfig:
    total_n_steps: int
    summary_freq: int
    agent_save_freq: int | None = None
    inference_freq: int | None = None

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
        self._inference_enabled = False
        
        self._time_steps = 0
        self._episodes = 0
        self._episode_len = 0
        self._real_start_time = time.time()
        self._real_time = 0.0
        
        self._cumulative_reward_mean = IncrementalAverage()
        self._episode_len_mean = IncrementalAverage()
        
    def enable_inference(self, infer_env: Env) -> "Train":
        if infer_env.num_envs != 1:
            raise Warning("it is recommended to use only one environment for inference")
        self._inference_enabled = True
        self._infer_env = infer_env
        return self
        
    def train(self):
        self._print_train_info()
        
        obs = self._env.reset().transform(self._agent_tensor)
        for t in range(self._time_steps, self._config.total_n_steps):
            # take action and observe
            action = self._agent.select_action(obs)
            next_obs, reward, terminated, real_final_next_obs = self._env.step(action)
            
            # update the agent
            real_next_obs = next_obs if real_final_next_obs is None else real_final_next_obs
            exp = Experience(
                obs,
                action,
                real_next_obs.transform(self._agent_tensor),
                self._agent_tensor(reward),
                self._agent_tensor(terminated),
            )
            self._agent.update(exp)
            
            # take next step
            obs = next_obs
            
            # summary
            if t % self._config.summary_freq == 0:
                self._summary_train()
                
            # inference
            if self._config.inference_freq is not None and t % self._config.inference_freq == 0:
                try:
                    self.inference()
                except InferenceError:
                    pass
    
    def inference(self):
        if not self._inference_enabled:
            raise InferenceError("inference is not enabled")
        
        with self._agent.behavior_type_scope(BehaviorType.INFERENCE):
            obs = self._infer_env.reset().transform(self._agent_tensor)
            not_terminated = True
            while not_terminated:
                # take action and observe
                action = self._agent.select_action(obs)
                next_obs, reward, terminated, real_final_next_obs = self._infer_env.step(action)
                
                # update the agent
                real_next_obs = next_obs if real_final_next_obs is None else real_final_next_obs
                exp = Experience(
                    obs,
                    action,
                    real_next_obs.transform(self._agent_tensor),
                    self._agent_tensor(reward),
                    self._agent_tensor(terminated),
                )
                self._agent.update(exp)
                
                # take next step
                obs = next_obs
                not_terminated = not terminated[0].item()
    
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
            .add_text(f"    total time steps: {self._config.total_n_steps}") \
            .add_text(f"    summary frequency: {self._config.summary_freq}") \
            .add_text(f"    agent save frequency: {self._config.agent_save_freq}") \
            .add_text(f"    inference frequency: {self._config.inference_freq}") \
            .add_line() \
            .add_text(f"Agent INFO:") \
            .add_text(f"    name: {self._agent.name}") \
            .add_text(f"    device: {self._agent.device}") \
            .make()
        
        logger.print(text_info_box, prefix="")
        logger.print("", prefix="")
        
    def _summary_train(self):
        if self._cumulative_reward_mean.count == 0:
            reward_info = "episode has not terminated yet"
        else:
            reward_info = f"cumulated reward: {self._cumulative_reward_mean.average:.2f}"
            logger.log("Environment/Cumulative Reward", self._cumulative_reward_mean.average, self._time_steps)
            logger.log("Environment/Episode Length", self._episode_len_mean.average, self._time_steps)
            self._cumulative_reward_mean.reset()
            self._episode_len_mean.reset()
        logger.print(f"training time: {self._real_time:.2f}, time steps: {self._time_steps}, {reward_info}")
        
        for key, (value, t) in self._agent.log_data:
            logger.log(key, value, t)