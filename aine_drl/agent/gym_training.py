from argparse import ArgumentTypeError
from gym import Env
from gym.vector import VectorEnv
from typing import Union
from aine_drl.agent import Agent
from aine_drl.drl_util import Experience
import aine_drl
import aine_drl.util as util

class GymTraining:
    """ It's a gym training class. """
    def __init__(self, env: Union[Env, VectorEnv], agent: Agent, env_id: str = None) -> None:
        if isinstance(env, VectorEnv):
            self.num_envs = env.num_envs
            self.is_vector_env = True
            self.env_id = env.get_attr("spec")[0].id if env_id is None else env_id
        elif isinstance(env, Env):
            self.num_envs = 1
            self.is_vector_env = False
            self.env_id = env.spec.id if env_id is None else env_id
        else:
            raise ArgumentTypeError("It isn't a gym environment.")
        assert env.new_step_api, "You must set new_step_api of the gym environment to True."
        
        self.env = env
        self.agent = agent
        if aine_drl.get_env_id() == "":
            aine_drl.set_env_id(self.env_id)
        
    def train(self, total_training_step: int, start_step: int = 0):
        """Start training.

        Args:
            total_training_step (int): total time step
            start_step (int, optional): training start step. Defaults to 0.
        """
        try:
            util.set_logger()
            self._train(total_training_step, start_step)
        finally:
            util.close_logger()
    
    def _train(self, total_training_step: int, start_step: int = 0):
        env = self.env
        states = env.reset()
        for _ in range(start_step, total_training_step, self.num_envs):
            actions = self.agent.act(states)
            # take action and observe
            next_states, rewards, terminateds, truncateds, _ = self.env.step(actions)
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
            states = next_states
