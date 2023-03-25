from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Callable

import torch

import aine_drl.policy.policy_dist as pd
from aine_drl.drl_util import Clock, Decay, ILogable, NoDecay


@dataclass(frozen=True)
class PolicyDistParam:
    """
    Standard policy distribution parameter (`pdparam`) data type.
    Note that these pdparams must be valid to the policy you currently use.
    
    When the action type is discrete, it is generally either logits or soft-max distribution. \\
    When the action type is continuous, it generally constitutes of both mean and standard deviation of gaussian distribution.
    
    `*batch_shape` depends on the input of the algorithm you are using.
    
    * simple batch: `*batch_shape` = `(batch_size,)`
    * sequence batch: `*batch_shape` = `(seq_batch_size, seq_len)`

    Args:
        discrete_pdparams (tuple[Tensor, ...]): `(*batch_shape, *discrete_pdparam_shape)` * `num_discrete_branches`
        continuous_pdparams (tuple[Tensor, ...]): `(*batch_shape, *continuous_pdparam_shape)` * `num_continuous_branches`
    """
    discrete_pdparams: tuple[torch.Tensor, ...] = field(default_factory=tuple)
    continuous_pdparams: tuple[torch.Tensor, ...] = field(default_factory=tuple)
    
    @property
    def num_discrete_branches(self) -> int:
        """Number of discrete action branches."""
        return len(self.discrete_pdparams)
    
    @property
    def num_continuous_branches(self) -> int:
        """Number of continuous action branches."""
        return len(self.continuous_pdparams)
    
    @property
    def num_branches(self) -> int:
        """Number of total branches."""
        return self.num_discrete_branches + self.num_continuous_branches
    
    def transform(self, func: Callable[[torch.Tensor], torch.Tensor]) -> "PolicyDistParam":
        """
        Transform each distribution parameter by the given function.
        """
        return PolicyDistParam(
            tuple(func(pdparam) for pdparam in self.discrete_pdparams),
            tuple(func(pdparam) for pdparam in self.continuous_pdparams)
        )

class ActionType(Flag):
    DISCRETE = auto()
    CONTINUOUS = auto()
    BOTH = DISCRETE | CONTINUOUS

class Policy(ABC):
    """
    Policy abstract class. It creates policy distribution.
    """
    @property
    @abstractmethod
    def action_type(self) -> ActionType:
        raise NotImplementedError
    
    @abstractmethod
    def policy_dist(self, pdparam: PolicyDistParam) -> pd.PolicyDist:
        """
        Create a policy distribution from `pdparam`.

        Args:
            pdparam (PolicyDistParam): policy distribution parameter which is generally the output of neural network

        Returns:
            policy_dist (PolicyDist): policy distribution or action probability distribution
        """
        raise NotImplementedError

class CategoricalPolicy(Policy):
    """
    Categorical policy for the discrete action type.
    
    Args:
        is_logit (bool, optional): wheter discrete parameters are logits or probabilities. Defaults to logit.
    """
    def __init__(self, is_logit: bool = True) -> None:
        self._is_logit = is_logit
        
    @property
    def action_type(self) -> ActionType:
        return ActionType.DISCRETE
        
    def policy_dist(self, pdparam: PolicyDistParam) -> pd.PolicyDist:
        if pdparam.num_discrete_branches == 0:
            raise PolicyDistParmBranchError(pdparam)
        
        if self._is_logit:
            return pd.CategoricalDist(logits=pdparam.discrete_pdparams)
        else:
            return pd.CategoricalDist(probs=pdparam.discrete_pdparams)
    
class GaussianPolicy(Policy):
    """
    Gaussian policy for the continuous action type.
    """
    @property
    def action_type(self) -> ActionType:
        return ActionType.CONTINUOUS
    
    def policy_dist(self, pdparam: PolicyDistParam) -> pd.PolicyDist:
        if pdparam.num_continuous_branches == 0:
            raise PolicyDistParmBranchError(pdparam)
        
        mean, std = GaussianPolicy.to_mean_std(pdparam.continuous_pdparams)
        return pd.GaussianDist(mean, std)
    
    @staticmethod
    def to_mean_std(continuous_pdparams: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        mean_params = []
        std_params = []
        
        for param in continuous_pdparams:
            mean_params.append(param[..., 0])
            std_params.append(param[..., 1])
            
        mean = torch.stack(mean_params, dim=-1)
        std = torch.stack(std_params, dim=-1) + 1e-8
        
        return mean, std

class CategoricalGaussianPolicy(Policy):
    """
    Categorical-Gaussian policy for the both discrete and continuous action type.
    
    Args:
        is_logit (bool, optional): wheter discrete parameters are logits or probabilities. Defaults to logit.
    """
    def __init__(self, is_logit: bool = True) -> None:
        self._is_logit = is_logit
    
    @property
    def action_type(self) -> ActionType:
        return ActionType.BOTH
    
    def policy_dist(self, pdparam: PolicyDistParam) -> pd.PolicyDist:
        if pdparam.discrete_pdparams == 0 or pdparam.continuous_pdparams == 0:
            raise PolicyDistParmBranchError(pdparam)
        
        mean, std = GaussianPolicy.to_mean_std(pdparam.continuous_pdparams)
        
        if self._is_logit:
            return pd.CategoricalGaussianDist(mean, std, logits=pdparam.discrete_pdparams)
        else:
            return pd.CategoricalGaussianDist(mean, std, probs=pdparam.discrete_pdparams)

class EpsilonGreedyPolicy(Policy):
    """
    Epsilon-greedy policy for value-based method. It only works to the discrete action type.
    
    Args:
        epsilon_decay (float | Decay): epsilon numerical value or decay instance. 0 <= epsilon <= 1
    """
    def __init__(self, epsilon_decay: float | Decay) -> None:
        if type(epsilon_decay) is float:
            epsilon_decay = NoDecay(epsilon_decay)
        
        self.epsilon_decay: Decay = epsilon_decay # type: ignore
        self.clock: Clock = None # type: ignore
        
    @property
    def action_type(self) -> ActionType:
        return ActionType.DISCRETE
        
    def policy_dist(self, pdparam: PolicyDistParam) -> pd.PolicyDist:
        if pdparam.num_discrete_branches == 0:
            raise PolicyDistParmBranchError(pdparam)
        
        return pd.EpsilonGreedyDist(pdparam.discrete_pdparams, self.epsilon_decay(0))

class BoltzmannPolicy(Policy):
    """
    TODO: implement
    """
    def __init__(self) -> None:
        raise NotImplementedError

class PolicyDistParmBranchError(ValueError):
    def __init__(self, pdparam: PolicyDistParam) -> None:
        if pdparam.num_branches == 0:
            message = f"you must specify at least one action branch. "
        elif pdparam.num_discrete_branches == 0:
            message = f"the policy requires discrete action branch. "
        else:
            message = f"the policy requires continuous action branch. "
        message += f"there are {pdparam.num_discrete_branches} discrete and {pdparam.num_continuous_branches} continuous branches."
        super().__init__(message)

class PolicyActionTypeError(TypeError):
    def __init__(self, valid_action_type: ActionType, invalid_policy: Policy) -> None:
        message = f"The policy action type must be \"{valid_action_type}\", but \"{type(invalid_policy).__name__}\" is \"{invalid_policy.action_type}\"."
        super().__init__(message)
