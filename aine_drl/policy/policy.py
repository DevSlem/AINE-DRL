from abc import ABC, abstractmethod
from typing import Union
import aine_drl.policy.policy_distribution as pd
from aine_drl.drl_util import Decay, NoDecay, Clock
from aine_drl.util.data_dict_provider import DataDictProvider

class Policy(DataDictProvider, ABC):
    """
    Policy abstract class. It returns policy distribution.
    """
    
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
        
    def get_policy_distribution(self, pdparam: pd.PolicyDistributionParameter) -> pd.PolicyDistribution:
        return pd.CategoricalDistribution(pdparam, self.is_logits)
    
class GaussianPolicy(Policy):
    """
    Gaussian policy for the continuous action type.
    """
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
    
    def get_policy_distribution(self, pdparam: pd.PolicyDistributionParameter) -> pd.PolicyDistribution:
        return pd.GeneralPolicyDistribution(pdparam, self.is_logits)

class EpsilonGreedyPolicy(Policy):
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
        
    def set_clock(self, clock: Clock):
        """
        Set clock to use epsilon decay.
        """
        self.clock = clock
        
    def get_policy_distribution(self, pdparam: pd.PolicyDistributionParameter) -> pd.PolicyDistribution:
        t = self.clock.global_time_step if self.clock is not None else 0
        return pd.EpsilonGreedyDistribution(pdparam, self.epsilon_decay(t))

class BoltzmannPolicy(Policy):
    """
    TODO: implement
    """
    pass
