from aine_drl.policy import Policy
from torch.distributions import Distribution, Categorical
import torch
from aine_drl.util.decorator import aine_api

class CategoricalPolicy(Policy):
    """
    Simple categorical policy.
    """
    
    @aine_api
    def get_policy_distribution(self, pdparam: torch.Tensor) -> Distribution:
        return Categorical(logits=pdparam)
