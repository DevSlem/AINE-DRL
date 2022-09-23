from aine_drl.policy import Policy
import torch
from torch.distributions import Distribution, Normal

class NormalPolicy(Policy):
    """
    Noraml distribution policy for the continuous action.
    """
    
    def get_policy_distribution(self, pdparam: torch.Tensor) -> Distribution:
        assert pdparam.shape[-1] == 2
        return Normal(loc=pdparam[..., 0], scale=pdparam[..., 1])
