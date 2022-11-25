from typing import Optional, Union
import yaml
from .agent import make_agent
from .agent.agent import Agent
from .network import Network
from .policy.policy import Policy
from .training.gym_training import GymTraining
from .training.gym_action_communicator import GymActionCommunicator
from gym import Env
from gym.vector import VectorEnv

class AINEConfig:
    def __init__(self, config_dir: str) -> None:
        self._config = AINEConfig.import_config(config_dir)
        
        keys = self._config.keys()
        assert len(keys) == 1
        self._training_env_id = list(keys)[0]
        self._num_envs = self.training_env_config["num_envs"]
        
    @property
    def training_env_id(self) -> str:
        return self._training_env_id
        
    @property
    def training_env_config(self) -> dict:
        return self._config[self.training_env_id]
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    def make_gym_training(self,
                          gym_env: Union[Env, VectorEnv, None] = None,
                          gym_action_communicator: Optional[GymActionCommunicator] = None) -> GymTraining:
        """
        ## Summary
        
        Make gym training instance from the YAML configuration.

        Args:
            gym_env (Env | VectorEnv | None, optional): gym environment. Defaults to making it from the configuration.
            gym_action_communicator (GymActionCommunicator | None, optional): action communicator between AINE-DRL and Gym. Defaults to auto set.

        Raises:
            ValueError: When you use custom `gym_env`, `num_envs` in the configuration must equal to the number of environments in `gym_env`.

        Returns:
            GymTraining: `GymTraining` instance
            
        ## Example
            
        YAML format::
        
            Gym:
              env:
                id: "CartPole-v1"
              training:
                total_global_time_steps: 100000
                summary_freq: 1000
                agent_save_freq: null
                inference_freq: 10000
                inference_render: true
                auto_retrain: false
                seed: 0
        """
        if isinstance(gym_env, VectorEnv):
            if self.num_envs != gym_env.num_envs:
                raise ValueError("When you use custom gym_env and the type is VectorEnv, num_envs in the configuration must equal to the number of environments in gym_env.")
        elif isinstance(gym_env, Env):
            if self.num_envs != 1:
                raise ValueError("When you use custom gym_env and the type is Env, num_envs in the configuration must equal to 1.")
            
        return GymTraining.make(
            self.training_env_id,
            self.training_env_config,
            self.num_envs,
            gym_env,
            gym_action_communicator
        )
    
    def make_agent(self, network: Network, policy: Policy) -> Agent:
        return make_agent(self.training_env_config, network, policy, self.num_envs)
        
    @staticmethod
    def import_config(config_dir: str) -> dict:
        with open(config_dir) as f:
            config = yaml.load(f, yaml.FullLoader)
        
        return config