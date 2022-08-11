from aine_drl.policy import Policy
from torch.distributions import Distribution, Categorical
import torch

class CategoricalPolicy(Policy):
    """
    Simple categorical policy.
    """
        
    def get_policy_distribution(pdparam: torch.Tensor) -> Distribution:
        return Categorical(logits=pdparam)
