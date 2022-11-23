from .agent import *
# DRL Agents
from .dqn import *
from .reinforce import *
from .a2c import *
from .ppo import *

# agent utility

from ..network import Network
from ..policy import Policy

def make_agent(agent_config: dict, network: Network, policy: Policy, num_envs: int) -> Agent:
    """
    ## Summary
    
    Make agents in `env_config`. The agent key must be the agent name like `PPO`. 
    In `env_config`, at least one agent key must be included. 
    The `network` type must be the type specified for each agent. 
    For example, the network type of `PPO` agent is `ActorCriticSharedNetwork`.

    Args:
        env_config (dict): environment configuration
        network (Network): network
        policy (Policy): policy
        num_envs (int): number of environments

    Returns:
        Agent: agent instance
        
    ## Example
    
    `env_config` dictionary format::
    
        {'PPO': {'training_freq': 16,
          'epoch': 3,
          'mini_batch_size': 8,
          'gamma': 0.99,
          'lam': 0.95,
          'epsilon_clip': 0.2,
          'value_loss_coef': 0.5,
          'entropy_coef': 0.001,
          'grad_clip_max_norm': 5.0}}}
    """
    
    try:
        
        for agent_key, config in agent_config.keys():
            if agent_key == "REINFORCE":
                if num_envs != 1:
                    raise ValueError(f"num_envs value must be 1, but {num_envs}.")
                config = REINFORCEConfig(**config)
                return REINFORCE(config, network, policy)  # type: ignore
            elif agent_key == "A2C":
                config = A2CConfig(**config)
                return A2C(config, network, policy, num_envs)  # type: ignore
            elif agent_key == "PPO":
                config = PPOConfig(**config)
                return PPO(config, network, policy, num_envs)  # type: ignore
            elif agent_key == "RecurrentPPO":
                config = RecurrentPPOConfig(**config)
                return RecurrentPPO(config, network, policy, num_envs)  # type: ignore
            elif agent_key == "DoubleDQN":
                config = DoubleDQNConfig(**config)
                return DoubleDQN(config, network, policy, num_envs)  # type: ignore
        
        raise ValueError("There's no agent configuration.")
    except Exception as e:
        raise e
