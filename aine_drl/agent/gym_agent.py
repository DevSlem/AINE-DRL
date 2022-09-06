from argparse import ArgumentTypeError
from gym import Env
from gym.vector import VectorEnv
from typing import Union
from aine_drl.agent import Agent
from aine_drl.drl_algorithm import DRLAlgorithm
from aine_drl.policy import Policy
from aine_drl.trajectory import Trajectory
from aine_drl.drl_util import Experience, Clock
from aine_drl import get_global_env_id, set_global_env_id
import aine_drl.util as util
from aine_drl.util.decorator import aine_api

class GymAgent(Agent):
    """ Gym agent class. """
    def __init__(self, 
                 gym_env: Union[Env, VectorEnv],
                 drl_algorithm: DRLAlgorithm, 
                 policy: Policy, 
                 trajectory: Trajectory, 
                 clock: Clock, 
                 summary_freq: int = 10,
                 env_id: str = None) -> None:
        """
        Gym agent class.

        Args:
            gym_env (Union[Env, VectorEnv]): gym environment
            drl_algorithm (DRLAlgorithm): DRL algorithm for agent training
            policy (Policy): policy to sample actions
            trajectory (Trajectory): trajectory to sample training batches
            clock (Clock): time step checker
            summary_freq (int, optional): summary frequency to log data. Defaults to 10.
            env_id (str, optional): custom environment id. Defaults to `gym_env` id.
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
            
        super().__init__(drl_algorithm, policy, trajectory, clock, summary_freq)
        
    @aine_api
    def train(self, total_training_step: int, start_step: int = 0):
        try:
            util.set_logger()
            self._train(total_training_step, start_step)
        finally:
            util.close_logger()
    
    def _train(self, total_training_step: int, start_step: int = 0):
        gym_env = self.gym_env
        states = gym_env.reset()
        for _ in range(start_step, total_training_step, self.num_envs):
            actions = self.act(states)
            # take action and observe
            next_states, rewards, terminateds, truncateds, _ = self.gym_env.step(actions)
            terminateds = terminateds | truncateds
            # update the agent
            if self.is_vector_env:
                exp_list = self.create_experience_list(
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
            self.update(exp_list)
            # update states
            states = next_states
