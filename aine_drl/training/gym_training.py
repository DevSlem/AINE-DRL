from argparse import ArgumentTypeError
from gym import Env
from gym.vector import VectorEnv
from typing import List, Union
from aine_drl.agent import Agent
from aine_drl.drl_util import Experience
from aine_drl import get_global_env_id, set_global_env_id
import aine_drl.util as util
from aine_drl.util import logger
import torch

class GymTraining:
    """ Gym training class. """
    def __init__(self,
                 agent: Agent,
                 gym_env: Union[Env, VectorEnv],
                 env_id: Union[str, None] = None,
                 seed: Union[int, List[int], None] = None,
                 auto_retrain: bool = False) -> None:
        """
        Gym training class.

        Args:
            agent (Agent): DRL agent to train
            gym_env (Union[Env, VectorEnv]): gym environment
            env_id (str, optional): custom environment id. Defaults to `gym_env` id.
            seed (Union[int, List[int], None], optional): gym environment random seed. if it's None, checks global random seed.
            auto_retrain (bool, optional): enable to retrain the agent. if you want to do, you must set custom `env_id`. Defaults to False.
        """
        
        if env_id is None and auto_retrain == True:
            raise ValueError("You must set auto_retrain to False when env_id is None.")
        
        self.auto_retrain = auto_retrain
        self.gym_env = gym_env
        
        if isinstance(gym_env, VectorEnv):
            self.num_envs = gym_env.num_envs
            self.is_vector_env = True
        elif isinstance(gym_env, Env):
            self.num_envs = 1
            self.is_vector_env = False
        else:
            raise ArgumentTypeError(f"You've instantiated GymTraining with {type(self.gym_env)} which isn't gym environment. Reinstantiate it.")
        
        assert gym_env.new_step_api, "You must set new_step_api of the gym environment to True."
        
        self.seed = seed if seed is not None else util.get_seed()
        self.agent = agent
        self._set_env_id(env_id)
        
    def run_train(self, total_time_steps: int):
        try:
            self._load_agent()
            self._train(total_time_steps)
        except KeyboardInterrupt:
            logger.print(f"Training interrupted at the time step {self.agent.clock.time_step}.")
        finally:
            logger.close()
            self._save_agent()
            
    def _set_env_id(self, env_id: Union[str, None]):
        """ set environment id. """
        if env_id is None:
            env_id = self.gym_env.get_attr("spec")[0].id if self.is_vector_env else self.gym_env.spec.id
        self.env_id = self._convert_env_id(env_id) if not self.auto_retrain else env_id
        set_global_env_id(self.env_id)
        
    def _convert_env_id(self, env_id: str) -> str:
        """ If the result of the `env_id` already exists, add a number suffix to the `env_id`. """
        dir = f"{logger.log_base_dir()}/{env_id}"
        if util.exists_dir(dir):
            dir = util.add_dir_num_suffix(dir, num_left="_")
            env_id = dir.replace(f"{logger.log_base_dir()}/", "", 1)
        return env_id
            
    def _save_agent(self):
        try:
            torch.save(self.agent.state_dict, logger.agent_save_dir())
        except FileNotFoundError:
            pass
            
    def _load_agent(self):
        if self.auto_retrain:
            try:
                ckpt = torch.load(logger.agent_save_dir())
                self.agent.load_state_dict(ckpt)
            except FileNotFoundError:
                pass
    
    def _train(self, total_time_steps: int):
        gym_env = self.gym_env
        states = gym_env.reset(seed=self.seed)
        if self.agent.clock.time_step >= total_time_steps:
            logger.print(f"Since {self.env_id} agent already reached to the total time steps, you can't train the agent.")
        for _ in range(self.agent.clock.time_step, total_time_steps, self.num_envs):
            actions = self.agent.select_action(states)
            # take action and observe
            next_states, rewards, terminateds, truncateds, _ = self.gym_env.step(actions)
            terminateds = terminateds | truncateds
            # update the agent
            if self.is_vector_env:
                exp_list = self.agent.create_experience_list(
                    states,
                    actions,
                    next_states,
                    rewards,
                    terminateds
                )
            else:
                exp_list = [Experience(
                    states,
                    actions,
                    next_states,
                    rewards,
                    terminateds
                )]
            self.agent.update(exp_list)
            # update states
            if (not self.is_vector_env) and terminateds:
                states = gym_env.reset(seed=self.seed)
            else:
                states = next_states
