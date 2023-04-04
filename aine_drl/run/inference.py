from __future__ import annotations
from dataclasses import dataclass

import torch
from PIL import Image

from aine_drl.agent.agent import Agent, BehaviorScope, BehaviorType
from aine_drl.exp import Experience
from aine_drl.util.func import create_dir
from aine_drl.util.logger import logger

from aine_drl.env import Env, Renderable
from aine_drl.run.error import AgentLoadError

class InferenceError(Exception):
    pass

@dataclass
class InferenceConfig:
    episodes: int
    export: str | None = "render_only"
    gif_duration: float = 33.0
    agent_file_path: str | None = None
    
class Inference:
    def __init__(
        self,
        id: str,
        config: InferenceConfig,
        env: Env,
        agent: Agent
    ) -> None:
        self._id = id
        self._config = config
        self._env = env
        self._renderable_env = env if isinstance(env, Renderable) else None
        self._frame_collection = []
        self._agent = agent
        
        self._dtype = torch.float32
        self._device = self._agent.device
        self._trace_env = 0
        
        self._enabled = True
        
    @property
    def config(self) -> InferenceConfig:
        return self._config
    
    @config.setter
    def config(self, config: InferenceConfig):
        self._config = config
        
    def inference(self) -> "Inference":
        if not self._enabled:
            raise RuntimeError("Inference is already closed.")
        
        with BehaviorScope(self._agent, BehaviorType.INFERENCE):
            if not logger.enabled():
                logger.enable(self._id, enable_log_file=False)
                
            self._load_inference()
            
            logger.disable()
            logger.enable(self._id, enable_log_file=False)
            
            try:
                for e in range(self._config.episodes):
                    obs = self._env.reset().transform(self._agent_tensor)
                    self._try_render()
                    not_terminated = True
                    cumulative_reward = 0.0
                    while not_terminated:
                        # take action and observe
                        action = self._agent.select_action(obs)
                        next_obs, reward, terminated, real_final_next_obs = self._env.step(action)
                        self._try_render()
                        
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
                        not_terminated = not terminated[self._trace_env].item()
                        cumulative_reward += reward[self._trace_env].item()

                    self._export(e)
                    logger.print(f"inference - episode: {e}, cumulative reward: {cumulative_reward:.2f}")
                logger.print(f"Inference is finished.")
            except KeyboardInterrupt:
                logger.print(f"Inference interrupted.")
                
        return self
    
    def close(self):
        self._enabled = False
        self._env.close()
        if logger.enabled():
            logger.disable()
            
    def _agent_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self._device, dtype=self._dtype)
    
    def _try_render(self):
        if (self._config.export is None) or (self._renderable_env is None):
            return
        frame = self._renderable_env.render()
        if (self._config.export != "render_only") and (frame is not None):
            self._frame_collection.append(frame)
            
    def _export(self, episode: int):
        images = tuple(Image.fromarray(frame) for frame in self._frame_collection)
        export = self._config.export
        if export == "render_only":
            pass
        elif export == "gif":
            if self._renderable_env is None:
                raise InferenceError("renderable environment is required for gif export")
            exports_dir = f"{logger.log_dir()}/exports/gifs"
            create_dir(exports_dir)
            images[0].save(
                f"{exports_dir}/{self._id}-episode{episode}.gif",
                save_all=True,
                append_images=images[1:],
                optimize=True,
                duration=self._config.gif_duration,
                loop=0
            )
        elif export == "picture":
            if self._renderable_env is None:
                raise InferenceError("renderable environment is required for picture export")
            exports_dir = f"{logger.log_dir()}/exports/pictures/episode{episode}"
            create_dir(exports_dir)
            for i, image in enumerate(images):
                image.save(f"{exports_dir}/{self._id}-episode{episode}-({i}).png")
        else:
            raise NotImplementedError(f"export type {export} is not supported")

        self._frame_collection.clear()
        
    def _load_inference(self):
        try:
            agent_file_path = self._config.agent_file_path
            state_dict = logger.load_agent(agent_file_path)
            
            if agent_file_path is None:
                agent_file_path = logger.agent_save_path()
                
            logger.print(f"Agent is successfully loaded from: {agent_file_path}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"no saved agent found at {logger.log_dir()}")
        try:
            self._agent.load_state_dict(state_dict["agent"])
        except:
            raise AgentLoadError("the loaded agent is not compatible with the current agent")
