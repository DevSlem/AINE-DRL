from abc import ABC, abstractmethod
from typing import NamedTuple, List, Optional
from aine_drl.experience import ActionTensor
import torch
from torch.distributions import Categorical, Normal

class PolicyDistributionParameter(NamedTuple):
    """
    Standard policy distribution parameter (`pdparam`) data type.
    Note that these pdparams must be valid to the policy you currently use. \\
    `batch_size` is `num_envs` x `n_steps`. \\
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
    
    @staticmethod
    def create(discrete_pdparams: Optional[List[torch.Tensor]] = None,
               continuous_pdparams: Optional[List[torch.Tensor]] = None) -> "PolicyDistributionParameter":
        if discrete_pdparams is None:
            discrete_pdparams = []
        if continuous_pdparams is None:
            continuous_pdparams = []
        
        return PolicyDistributionParameter(discrete_pdparams, continuous_pdparams)


class PolicyDistribution(ABC):
    """Policy distribution abstract class. It provides the policy utility like action selection."""
    
    @abstractmethod
    def sample(self) -> ActionTensor:
        """Sample the action from the policy distribution."""
        raise NotImplementedError
    
    @abstractmethod
    def log_prob(self, action: ActionTensor) -> torch.Tensor:
        """
        Returns the log of the porability density function evaluated at the `action`. 
        The returned log probability shape is `(batch_size, num_branches)`.
        """
        raise NotImplementedError
    
    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """Returns entropy of distribution. The returned entropy shape is `(batch_size, num_branches)`."""
        raise NotImplementedError
    

class CategoricalDistribution(PolicyDistribution):
    """Categorical policy distribution for the discrete action type."""
    def __init__(self, pdparam: PolicyDistributionParameter, is_logits: bool = True) -> None:
        assert pdparam.num_discrete_branches > 0
        self.distributions = []
        if is_logits:
            for param in pdparam.discrete_pdparams:
                self.distributions.append(Categorical(logits=param))
        else:
            for param in pdparam.discrete_pdparams:
                self.distributions.append(Categorical(probs=param))
        
    def sample(self) -> ActionTensor:
        sampled_discrete_action = []
        for dist in self.distributions:
            sampled_discrete_action.append(dist.sample())
        sampled_discrete_action = torch.stack(sampled_discrete_action, dim=1)
        return ActionTensor.create(sampled_discrete_action, None)
    
    def log_prob(self, action: ActionTensor) -> torch.Tensor:
        action_log_prob = []
        for i, dist in enumerate(self.distributions):
            # TODO: Error 발생
            action_log_prob.append(dist.log_prob(action.discrete_action[:, i]))
        return torch.stack(action_log_prob, dim=1)
    
    def entropy(self) -> torch.Tensor:
        entropies = []
        for dist in self.distributions:
            entropies.append(dist.entropy())
        return torch.stack(entropies, dim=1)
    

class GaussianDistribution(PolicyDistribution):
    """Gaussian policy distribution for the continuous action type."""
    def __init__(self, pdparam: PolicyDistributionParameter) -> None:
        assert pdparam.num_continuous_branches > 0
        self.distributions = []
        for param in pdparam.continuous_pdparams:
            self.distributions.append(Normal(loc=param[:, 0], scale=param[:, 1]))
            
    def sample(self) -> ActionTensor:
        sampled_continuous_action = []
        for dist in self.distributions:
            sampled_continuous_action.append(dist.sample())
        sampled_continuous_action = torch.stack(sampled_continuous_action, dim=1)
        return ActionTensor.create(None, sampled_continuous_action)
    
    def log_prob(self, action: ActionTensor) -> torch.Tensor:
        action_log_prob = []
        for i, dist in enumerate(self.distributions):
            action_log_prob.append(dist.log_prob(action.continuous_action[:, i]))
        return torch.stack(action_log_prob, dim=1)
    
    def entropy(self) -> torch.Tensor:
        entropies = []
        for dist in self.distributions:
            entropies.append(dist.entropy())
        return torch.stack(entropies, dim=1)


class GeneralPolicyDistribution(PolicyDistribution):
    """
    General policy distribution for the both discrete and continuous action type. \\
    Policy distribution of the discrete action type is categorical. \\
    Policy distribution of the continuous action type is gaussian.
    """
    def __init__(self, pdparam: PolicyDistributionParameter, is_logits: bool = True) -> None:
        self.discrete_dist = CategoricalDistribution(pdparam, is_logits)
        self.continuous_dist = GaussianDistribution(pdparam)
        
    def sample(self) -> ActionTensor:
        discrete_action = self.discrete_dist.sample()
        continuous_action = self.continuous_dist.sample()
        return ActionTensor(discrete_action.discrete_action, continuous_action.continuous_action)
    
    def log_prob(self, action: ActionTensor) -> torch.Tensor:
        discrete_log_prob = self.discrete_dist.log_prob(action)
        continuous_log_prob = self.continuous_dist.log_prob(action)
        return torch.cat([discrete_log_prob, continuous_log_prob], dim=1)
    
    def entropy(self) -> torch.Tensor:
        discrete_entropy = self.discrete_dist.entropy()
        continuous_entropy = self.continuous_dist.entropy()
        return torch.cat([discrete_entropy, continuous_entropy], dim=1)


class EpsilonGreedyDistribution(CategoricalDistribution):
    """
    Epsilon-greedy policy distribution.
    """
    
    def __init__(self, pdparam: PolicyDistributionParameter, epsilon: float) -> None:
        pdparam = EpsilonGreedyDistribution.get_epsilon_greedy_pdparam(pdparam, epsilon)
        super().__init__(pdparam, False)
    
    @staticmethod
    def get_epsilon_greedy_pdparam(pdparam: PolicyDistributionParameter, epsilon: float) -> PolicyDistributionParameter:
        epsilon_greedy_probs = []
        
        # set epsilon greedy probability distribution
        for i in range(pdparam.num_discrete_branches):
            q_values = pdparam.discrete_pdparams[i]
            num_actions = q_values.shape[1]
            
            # epsilon-greedy probabilities
            greedy_action_prob = 1.0 - epsilon + epsilon / num_actions
            non_greedy_action_prob = epsilon / num_actions
            
            # get greedy action
            greedy_action = q_values.argmax(dim=1, keepdim=True)
            
            # set epsilon greedy probability distribution
            epsilon_greedy_prob = torch.full_like(q_values, non_greedy_action_prob)
            epsilon_greedy_prob.scatter(1, greedy_action, greedy_action_prob)
            epsilon_greedy_probs.append(epsilon_greedy_prob)
            
        return PolicyDistributionParameter.create(discrete_pdparams=epsilon_greedy_probs)
        