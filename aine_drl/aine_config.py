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
    """
    AINE-DRL configuration helper class. It parse YAML format configuration and makes training instance, agent, etc.
    """
    def __init__(self, config_dir: str) -> None:
        self._config = AINEConfig.import_config(config_dir)
        
        keys = self._config.keys()
        assert len(keys) == 1
        self._training_env_id = list(keys)[0]
        self._num_envs = self.training_env_config["num_envs"]
        
    @property
    def training_env_id(self) -> str:
        """Training environment id."""
        return self._training_env_id
        
    @property
    def training_env_config(self) -> dict:
        """Training environment configuration."""
        return self._config[self.training_env_id]
    
    @property
    def num_envs(self) -> int:
        """The number of environments specified in the configuration."""
        return self._num_envs
    
    def make_gym_training(self,
                          gym_env: Union[Env, VectorEnv, None] = None,
                          gym_action_communicator: Optional[GymActionCommunicator] = None) -> GymTraining:
        """
        ## Summary
        
        Make gym training instance from the YAML configuration. 
        The only main key is `Gym` and two sub keys are `env` which specify to make a gym environment and `training` which specify training configurations.
        Configurations of `env` key are parameters of `gym.make()` function. 
        Configurations of `training` key are fields of `GymTrainingConfig` class.

        Args:
            gym_env (Env | VectorEnv | None, optional): gym environment. Defaults to making it from the configuration.
            gym_action_communicator (GymActionCommunicator | None, optional): action communicator between AINE-DRL and Gym. Defaults to auto set.

        Raises:
            ValueError: When you use custom `gym_env` and the type is `VectorEnv`, `num_envs` in the configuration must equal to the number of environments in `gym_env`.
            ValueError: When you use custom `gym_env` and the type is `Env`, `num_envs` in the configuration must equal to `1`.

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
        """
        ## Summary
    
        Make an agent from the YAML configuration. 
        The main key must be the agent name like `PPO` and at least one agent key must be included. 
        See configuration details in AINE-DRL Wiki - https://github.com/DevSlem/AINE-DRL/wiki. \\
        The `network` type must be the type specified for each agent. 
        For example, the network type of `PPO` agent is `ActorCriticSharedNetwork`.
        
        network (Network): network
        policy (Policy): policy
        
        Returns:
            Agent: `Agent` instance
            
        ## Example
        
        YAML format::
        
            PPO:
              training_freq: 256
              epoch: 4
              mini_batch_size: 128
              gamma: 0.99
              lam: 0.95
              epsilon_clip: 0.2
              value_loss_coef: 0.2
              entropy_coef: 0.001
        """
        return make_agent(self.training_env_config, network, policy, self.num_envs)
        
    @staticmethod
    def import_config(config_dir: str) -> dict:
        with open(config_dir) as f:
            config = yaml.load(f, yaml.FullLoader)
        
        return config