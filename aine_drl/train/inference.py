from dataclasses import dataclass

import torch
from PIL import Image

from aine_drl.agent.agent import Agent, BehaviorScope, BehaviorType
from aine_drl.exp import Experience
from aine_drl.util.logger import logger

from .env import Env, Renderable
from .error import AgentLoadError


@dataclass
class InferenceConfig:
    episodes: int
    export: str | None = "render_only"
    
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
        
        if not logger.enabled():
            logger.enable(self._id)
        
        with BehaviorScope(self._agent, BehaviorType.INFERENCE):
            self._load_inference()
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
                    real_next_obs = next_obs if real_final_next_obs is None else real_final_next_obs.transform(self._agent_tensor)
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

                logger.print(f"inference - episode: {e}, cumulative reward: {cumulative_reward}")
                self._export(e)
                
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
        if (self._config.export == "render_only") and (frame is not None):
            self._frame_collection.append(frame)
            
    def _export(self, episode: int):
        images = tuple(Image.fromarray(frame) for frame in self._frame_collection)
        match self._config.export:
            case "render_only":
                pass
            case "gif":
                images[0].save(
                    f"{logger.log_dir()}/exports/gifs/{self._id}-episode{episode}.gif",
                    save_all=True,
                    append_images=images[1:],
                    optimize=False,
                    duration=40,
                    loop=0
                )
            case "picture":
                for i, image in enumerate(images):
                    image.save(f"{logger.log_dir()}/pictures/episode{episode}/{self._id}-episode{episode}-({i}).png")
            case "video":
                raise NotImplementedError("video export is not implemented yet")
            case _:
                raise NotImplementedError(f"export type {self._config.export} is not supported")
            
        self._frame_collection.clear()
        
    def _load_inference(self):
        try:
            state_dict = logger.load_agent()
        except FileNotFoundError:
            raise FileNotFoundError(f"no saved agent found at {logger.log_dir()}")
        try:
            self._agent.load_state_dict(state_dict["agent"])
        except:
            raise AgentLoadError("the loaded agent is not compatible with the current agent")
