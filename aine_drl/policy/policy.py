from abc import ABC, abstractmethod
import aine_drl.policy.policy_distribution as pd
import torch

class Policy(ABC):
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
        return pd.CategoricalPolicyDistribution(pdparam, self.is_logits)
    

class GaussianPolicy(Policy):
    """
    Gaussian policy for the continuous action type.
    """
    def get_policy_distribution(self, pdparam: pd.PolicyDistributionParameter) -> pd.PolicyDistribution:
        return pd.GaussianPolicyDistribution(pdparam)


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
    TODO: implement
    """
    pass
