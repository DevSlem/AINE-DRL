from typing import Optional, Union
import yaml
from aine_drl.agent import make_agent
from aine_drl.agent.agent import Agent
from aine_drl.network import Network
from aine_drl.policy.policy import Policy
from aine_drl.training.gym_training import GymTraining
from aine_drl.training.gym_action_communicator import GymActionCommunicator
from gym import Env
from gym.vector import VectorEnv

class ConfigManager:
    def __init__(self, config_dir: str) -> None:
        self._config = ConfigManager.import_config(config_dir)
        
        keys = self._config.keys()
        assert len(keys) == 1
        self._env_id = list(keys)[0]
        self._num_envs = self.env_config["num_envs"]
        
    @property
    def env_id(self) -> str:
        return self._env_id
        
    @property
    def env_config(self) -> dict:
        return self._config[self.env_id]
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    def make_gym_training(self,
                          env_id: Optional[str] = None,
                          gym_env: Union[Env, VectorEnv, None] = None,
                          gym_action_communicator: Optional[GymActionCommunicator] = None) -> GymTraining:
        return GymTraining.make(self.env_config, env_id=env_id, gym_env=gym_env, gym_action_communicator=gym_action_communicator)
    
    def make_agent(self, network: Network, policy: Policy) -> Agent:
        return make_agent(self.env_config, network, policy, self.num_envs)
        
    @staticmethod
    def import_config(config_dir: str) -> dict:
        with open(config_dir) as f:
            config = yaml.load(f, yaml.FullLoader)
        
        return config