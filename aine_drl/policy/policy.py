from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union
import aine_drl.policy.policy_distribution as pd
from aine_drl.drl_util import Decay, NoDecay, Clock, ILogable
from enum import Flag, auto

class ActionType(Flag):
    DISCRETE = auto()
    CONTINUOUS = auto()
    BOTH = DISCRETE | CONTINUOUS

class Policy(ABC):
    """
    Policy abstract class. It returns policy distribution.
    """
    
    @property
    @abstractmethod
    def action_type(self) -> ActionType:
        raise NotImplementedError
    
    @abstractmethod
    def get_policy_distribution(self, pdparam: pd.PolicyDistributionParameter) -> pd.PolicyDistribution:
        """
        Returns policy distribution which is action probability distribution.

        Args:
            pdparam (PolicyDistributionParameter): policy distribution parameter which is generally the output of neural network

        Returns:
            PolicyDistribution: policy distribution
        """
        raise NotImplementedError

class CategoricalPolicy(Policy):
    """
    Categorical policy for the discrete action type.
    
    Args:
        is_logits (bool, optional): wheter logits or probabilities mode. Defaults to logits.
    """
    def __init__(self, is_logits: bool = True) -> None:
        self.is_logits = is_logits
        
    @property
    def action_type(self) -> ActionType:
        return ActionType.DISCRETE
        
    def get_policy_distribution(self, pdparam: pd.PolicyDistributionParameter) -> pd.PolicyDistribution:
        return pd.CategoricalDistribution(pdparam, self.is_logits)
    
class GaussianPolicy(Policy):
    """
    Gaussian policy for the continuous action type.
    """
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.CONTINUOUS
    
    def get_policy_distribution(self, pdparam: pd.PolicyDistributionParameter) -> pd.PolicyDistribution:
        return pd.GaussianDistribution(pdparam)

class GeneralPolicy(Policy):
    """
    General policy for the both discrete and continuous action type.
    
    Args:
        is_logits (bool, optional): for the discrete action type, wheter logits or probabilities mode. Defaults to logits.
    """
    def __init__(self, is_logits: bool = True) -> None:
        self.is_logits = is_logits
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.BOTH
    
    def get_policy_distribution(self, pdparam: pd.PolicyDistributionParameter) -> pd.PolicyDistribution:
        return pd.GeneralPolicyDistribution(pdparam, self.is_logits)

class EpsilonGreedyPolicy(Policy, ILogable):
    """
    Epsilon-greedy policy for value-based method. It only works to the discrete action type.
    
    Args:
        epsilon_decay (float | Decay): epsilon numerical value or decay instance. 0 <= epsilon <= 1
    """
    def __init__(self, epsilon_decay: Union[float, Decay]) -> None:
        if type(epsilon_decay) is float:
            epsilon_decay = NoDecay(epsilon_decay)
        
        self.epsilon_decay = epsilon_decay
        self.clock = None
        
    @property
    def action_type(self) -> ActionType:
        return ActionType.DISCRETE
        
    def get_policy_distribution(self, pdparam: pd.PolicyDistributionParameter) -> pd.PolicyDistribution:
        return pd.EpsilonGreedyDistribution(pdparam, self.epsilon_decay(self.clock.global_time_step))
    
    def set_clock(self, clock: Clock):
        self.clock = clock
    
    @property
    def log_keys(self) -> Tuple[str, ...]:
        return ("Policy/Epsilon",)
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        t = self.clock.global_time_step
        return {"Policy/Epsilon": (self.epsilon_decay(t), t)}

class BoltzmannPolicy(Policy):
    """
    TODO: implement
    """
    pass
