from argparse import ArgumentTypeError
from gym import Env
from gym.vector import VectorEnv
from typing import List, Union
from aine_drl.agent import Agent
from aine_drl.drl_util import Experience
from aine_drl import get_global_env_id, set_global_env_id
import aine_drl.util as util

class GymTraining:
    """ Gym agent class. """
    def __init__(self,
                 agent: Agent,
                 gym_env: Union[Env, VectorEnv],
                 env_id: str = None,
                 seed: Union[int, List[int], None] = None) -> None:
        """
        Gym agent class.

        Args:
            agent (Agent): DRL Agent to train
            gym_env (Union[Env, VectorEnv]): gym environment
            env_id (str, optional): custom environment id. Defaults to `gym_env` id.
            seed (Union[int, List[int], None], optional): gym environment random seed. if it's None, checks global random seed.
        """
        if isinstance(gym_env, VectorEnv):
            self.num_envs = gym_env.num_envs
            self.is_vector_env = True
            self.env_id = gym_env.get_attr("spec")[0].id if env_id is None else env_id
        elif isinstance(gym_env, Env):
            self.num_envs = 1
            self.is_vector_env = False
            self.env_id = gym_env.spec.id if env_id is None else env_id
        else:
            raise ArgumentTypeError("It isn't a gym environment.")
        assert gym_env.new_step_api, "You must set new_step_api of the gym environment to True."
        
        self.gym_env = gym_env
        if get_global_env_id() == "":
            set_global_env_id(self.env_id)
        self.seed = seed if seed is not None else util.get_seed()
        self.agent = agent
        
    def run_train(self, total_time_steps: int, start_step: int = 0):
        try:
            util.set_logger()
            self._train(total_time_steps, start_step)
        finally:
            util.close_logger()
    
    def _train(self, total_time_steps: int, start_step: int = 0):
        gym_env = self.gym_env
        states = gym_env.reset(seed=self.seed)
        for _ in range(start_step, total_time_steps, self.num_envs):
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
