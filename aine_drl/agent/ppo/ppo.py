from typing import Union, NamedTuple, Optional
from aine_drl.agent import Agent
from aine_drl.drl_util.clock import Clock
from aine_drl.experience import Experience
from aine_drl.policy.policy import Policy
from aine_drl.network import ActorCriticSharedNetwork
from aine_drl.trajectory.batch_trajectory import BatchTrajectory
import torch

class PPOConfig(NamedTuple):
    """
    PPO configuration.

    Args:
        training_freq (int): start training when the number of calls of `Agent.update()` method is reached to training frequency
        epoch (int): training epoch
        mini_batch_size (int): sampled batch is randomly split into mini batch size
        gamma (float): discount factor. Defaults to 0.99.
        lam (float): lambda which controls the balanace of GAE between bias and variance. Defaults to 0.95.
        epsilon_clip (float): clipping the probability ratio (pi_theta / pi_theta_old) to [1-eps, 1+eps]. Defaults to 0.2.
        value_loss_coef (float, optional): value loss multiplier. Defaults to 0.5.
        entropy_coef (float, optional): entropy multiplier. Defaults to 0.001.
        grad_clip_max_norm (float | None, optional): gradient clipping maximum value. Defaults to no gradient clipping.
    """
    training_freq: int
    epoch: int
    mini_batch_size: int
    gamma: float = 0.99
    lam: float = 0.95
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    grad_clip_max_norm: Optional[float] = None
    
class PPO(Agent):
    """
    Proximal policy optimization (PPO). See details in https://arxiv.org/abs/1707.06347.
    """
    def __init__(self, 
                 config: PPOConfig,
                 network: ActorCriticSharedNetwork,
                 num_envs: int) -> None:
        """
        Proximal Policy Optimization (PPO). See details in https://arxiv.org/abs/1707.06347.

        Args:
            config (PPOConfig): ppo configuration
            network (ActorCriticSharedNetwork): standard actor critic network
        """
        super().__init__(num_envs)
        
        self.config = config
        self.network = network
        
    pass