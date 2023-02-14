from abc import ABC, abstractmethod
from typing import NamedTuple, List, Optional, Tuple, Callable
from aine_drl.experience import ActionTensor
import torch
from torch.distributions import Categorical, Normal
from dataclasses import dataclass, field

class PolicyDistributionParameter(NamedTuple):
    """
    Policy distribution parameter batch (`pdparam`) data type.
    Note that these `pdparam` must be valid to the policy you currently use.
    
    When the action type is discrete, it is generally either logits or soft-max distribution. \\
    When the action type is continuous, it is generally mean and standard deviation of gaussian distribution.

    Args:
        discrete_pdparams (List[Tensor]): `(batch_size, *discrete_pdparam_shape)` x `num_discrete_branches`
        continuous_pdparams (List[Tensor]): `(batch_size, *continuous_pdparam_shape)` x `num_continuous_branches`
    """
    discrete_pdparams: List[torch.Tensor]
    continuous_pdparams: List[torch.Tensor]
    
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
    
    def flattened_to_sequence(self, seq_len: int) -> "PolicyDistributionParameter":
        discrete_pdparam_sequences = []
        continuous_pdparam_sequences = []
        
        for pdparam_batch in self.discrete_pdparams:
            discrete_pdparam_shape = pdparam_batch.shape[1:]
            discrete_pdparam_sequences.append(pdparam_batch.reshape(-1, seq_len, *discrete_pdparam_shape))
            
        for pdparam_batch in self.continuous_pdparams:
            continuous_pdparam_shape = pdparam_batch.shape[1:]
            continuous_pdparam_sequences.append(pdparam_batch.reshape(-1, seq_len, *continuous_pdparam_shape))
            
        return PolicyDistributionParameter(discrete_pdparam_sequences, continuous_pdparam_sequences)
    
    def sequence_to_flattened(self) -> "PolicyDistributionParameter":
        discrete_pdaparam_batches = []
        continuous_pdaparam_batches = []
        
        for pdparam_seq in self.discrete_pdparams:
            discrete_pdaparam_batches.append(pdparam_seq.flatten(0, 1))
        
        for pdparam_seq in self.continuous_pdparams:
            continuous_pdaparam_batches.append(pdparam_seq.flatten(0, 1))
            
        return PolicyDistributionParameter(discrete_pdaparam_batches, continuous_pdaparam_batches)
    
    @staticmethod
    def new(discrete_pdparams: Optional[List[torch.Tensor]] = None,
               continuous_pdparams: Optional[List[torch.Tensor]] = None) -> "PolicyDistributionParameter":
        if discrete_pdparams is None:
            discrete_pdparams = []
        if continuous_pdparams is None:
            continuous_pdparams = []
        
        return PolicyDistributionParameter(discrete_pdparams, continuous_pdparams)
    
@dataclass(frozen=True)
class PolicyDistParam:
    """
    Standard policy distribution parameter (`pdparam`) data type.
    Note that these pdparams must be valid to the policy you currently use.
    
    When the action type is discrete, it is generally either logits or soft-max distribution. \\
    When the action type is continuous, it is generally mean and standard deviation of gaussian distribution.
    
    `*batch_shape` depends on the input of the algorithm you are using. \\
    If it's simple batch, `*batch_shape` = `(batch_size,)`. \\
    If it's sequence batch, `*batch_shape` = `(num_seq, seq_len)`.

    Args:
        discrete_pdparams (List[Tensor]): `(*batch_shape, *discrete_pdparam_shape)` x `num_discrete_branches`
        continuous_pdparams (List[Tensor]): `(*batch_shape, *continuous_pdparam_shape)` x `num_continuous_branches`
    """
    discrete_pdparams: Tuple[torch.Tensor] = field(default_factory=tuple)
    continuous_pdparams: Tuple[torch.Tensor] = field(default_factory=tuple)
    
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

class PolicyDistribution(ABC):
    """
    Policy distribution abstract class. 
    It provides the policy utility like action selection. 
    Each action branch is generally independent, 
    so its components of the log probability or the entropy are summed. 
    """
    
    @abstractmethod
    def sample(self) -> ActionTensor:
        """Sample the action from the policy distribution."""
        raise NotImplementedError
    
    @abstractmethod
    def log_prob(self, action: ActionTensor) -> torch.Tensor:
        """
        Returns the log of the porability mass/density function accroding to the `action`.
        
        Returns:
            log_prob (Tensor): `(*batch_shape, num_branches)`
        """
        raise NotImplementedError
    
    @abstractmethod
    def joint_log_prob(self, action: ActionTensor) -> torch.Tensor:
        """
        Returns the joint log of the porability mass/density function accroding to the `action`.
        
        Returns:
            joint_log_prob (Tensor): `(*batch_shape, 1)`
        """
    
    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """
        Returns the entropy of this distribution. 
        
        Returns:
            entropy (Tensor): `(*batch_shape, num_branches)`
        """
        raise NotImplementedError
    
    @abstractmethod
    def joint_entropy(self) -> torch.Tensor:
        """
        Returns the joint entropy of this distribution. 
        
        Returns:
            joint_entropy (Tensor): `(*batch_shape, 1)`
        """
        raise NotImplementedError
    

class CategoricalDistribution(PolicyDistribution):
    """Categorical policy distribution for the discrete action type."""
    def __init__(self, pdparam: PolicyDistParam, is_logits: bool = True) -> None:
        assert pdparam.num_discrete_branches > 0
        
        if is_logits:
            self.distributions = tuple(
                Categorical(logits=param) for param in pdparam.discrete_pdparams
            )
        else:
            self.distributions = tuple(
                Categorical(probs=param) for param in pdparam.discrete_pdparams
            )
        
    def sample(self) -> ActionTensor:
        sampled_discrete_action = [dist.sample() for dist in self.distributions]
        sampled_discrete_action = torch.stack(sampled_discrete_action, dim=-1)
        return ActionTensor(discrete_action=sampled_discrete_action)
    
    def log_prob(self, action: ActionTensor) -> torch.Tensor:
        action_log_prob = []
        for i, dist in enumerate(self.distributions):
            action_log_prob.append(dist.log_prob(action.discrete_action[..., i]))
        return torch.stack(action_log_prob, dim=-1)
    
    def joint_log_prob(self, action: ActionTensor) -> torch.Tensor:
        return self.log_prob(action).sum(dim=-1, keepdim=True)
    
    def entropy(self) -> torch.Tensor:
        entropies = [dist.entropy() for dist in self.distributions]
        return torch.stack(entropies, dim=-1)
    
    def joint_entropy(self) -> torch.Tensor:
        return self.entropy().sum(dim=-1, keepdim=True)

class GaussianDistribution(PolicyDistribution):
    """Gaussian policy distribution for the continuous action type."""
    
    LOG_STD_MIN: float = -20
    LOG_STD_MAX: float = 2
    
    def __init__(self, pdparam: PolicyDistParam, is_log_std: bool = True) -> None:
        assert pdparam.num_continuous_branches > 0
        
        mean_params = []
        std_params = []
        
        for param in pdparam.continuous_pdparams:
            mean_params.append(param[..., 0])
            std_params.append(param[..., 1])
            
        mean = torch.stack(mean_params, dim=-1)
        std_params = torch.stack(std_params, dim=-1)
        if is_log_std:
            log_std = torch.clamp(std_params, GaussianDistribution.LOG_STD_MIN, GaussianDistribution.LOG_STD_MAX)
            std = log_std.exp()
        else:
            std = std_params.abs()
        
        # (num_envs, n, n)
        # covars = torch.stack([torch.eye(pdparam.num_continuous_branches) for _ in range(len(var))]).to(device=var.device)
        # covars = var.unsqueeze(dim=-1) * covars
        
        # self.distributions = MultivariateNormal(mean, covars)
        
        self.distribution = Normal(mean, std)
        
            
    def sample(self) -> ActionTensor:
        return ActionTensor(continuous_action=self.distribution.rsample())
    
    def log_prob(self, action: ActionTensor) -> torch.Tensor:
        return self.distribution.log_prob(action.continuous_action)
    
    def joint_log_prob(self, action: ActionTensor) -> torch.Tensor:
        return self.log_prob(action).sum(dim=-1, keepdim=True)
    
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()
    
    def joint_entropy(self) -> torch.Tensor:
        return self.entropy().sum(dim=-1, keepdim=True)
    
class GeneralPolicyDistribution(PolicyDistribution):
    """
    General policy distribution for the both discrete and continuous action type. \\
    Policy distribution of the discrete action type is categorical. \\
    Policy distribution of the continuous action type is gaussian.
    """
    def __init__(self, 
                 pdparam: PolicyDistParam, 
                 is_logits: bool = True,
                 is_log_std: bool = True) -> None:
        self.discrete_dist = CategoricalDistribution(pdparam, is_logits)
        self.continuous_dist = GaussianDistribution(pdparam, is_log_std)
        
    def sample(self) -> ActionTensor:
        discrete_action = self.discrete_dist.sample()
        continuous_action = self.continuous_dist.sample()
        return ActionTensor(discrete_action.discrete_action, continuous_action.continuous_action)
    
    def log_prob(self, action: ActionTensor) -> torch.Tensor:
        discrete_log_prob = self.discrete_dist.log_prob(action)
        continuous_log_prob = self.continuous_dist.log_prob(action)
        return torch.cat((discrete_log_prob, continuous_log_prob), dim=-1)
    
    def joint_log_prob(self, action: ActionTensor) -> torch.Tensor:
        return self.log_prob(action).sum(dim=-1, keepdim=True)
    
    def entropy(self) -> torch.Tensor:
        discrete_entropy = self.discrete_dist.entropy()
        continuous_entropy = self.continuous_dist.entropy()
        return torch.cat((discrete_entropy, continuous_entropy), dim=-1)

    def joint_entropy(self) -> torch.Tensor:
        return self.entropy().sum(dim=-1, keepdim=True)

class EpsilonGreedyDistribution(CategoricalDistribution):
    """
    Epsilon-greedy policy distribution.
    """
    
    def __init__(self, pdparam: PolicyDistParam, epsilon: float) -> None:
        pdparam = EpsilonGreedyDistribution.get_epsilon_greedy_pdparam(pdparam, epsilon)
        super().__init__(pdparam, is_logits=False)
    
    @staticmethod
    def get_epsilon_greedy_pdparam(pdparam: PolicyDistParam, epsilon: float) -> PolicyDistParam:
        epsilon_greedy_probs = []
        
        # set epsilon greedy probability distribution
        for i in range(pdparam.num_discrete_branches):
            q_values = pdparam.discrete_pdparams[i]
            num_actions = q_values.shape[-1]
            
            # epsilon-greedy probabilities
            greedy_action_prob = 1.0 - epsilon + epsilon / num_actions
            non_greedy_action_prob = epsilon / num_actions
            
            # get greedy action
            greedy_action = q_values.argmax(dim=-1, keepdim=True)
            
            # set epsilon greedy probability distribution
            epsilon_greedy_prob = torch.full_like(q_values, non_greedy_action_prob)
            epsilon_greedy_prob.scatter_(-1, greedy_action, greedy_action_prob)
            epsilon_greedy_probs.append(epsilon_greedy_prob)
            
        return PolicyDistParam(discrete_pdparams=tuple(epsilon_greedy_probs))
        