from aine_drl.policy import Policy
from torch.distributions import Distribution, Categorical
import torch

class CategoricalPolicy(Policy):
    """
    Simple categorical policy for the discrete action.
    """
    
    def __init__(self, is_probs: bool = False) -> None:
        self.is_probs = is_probs
    
    def get_policy_distribution(self, pdparam: torch.Tensor) -> Distribution:
        if self.is_probs:
            return Categorical(probs=pdparam)
        else:
            return Categorical(logits=pdparam)
