from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torch.distributions as D

from aine_drl.exp import Action


class PolicyDist(ABC):
    """
    Policy distribution interface.
    
    `*batch_shape` depends on the input of the algorithm you are using.
    
    * simple batch: `*batch_shape` = `(batch_size,)`
    * sequence batch: `*batch_shape` = `(seq_batch_size, seq_len)`
    """
    @abstractmethod
    def sample(self, reparam_trick: bool = False) -> Action:
        """
        Sample action from the policy distribution.

        Args:
            reparam_trick (bool, optional): reparameterization trick. Defaults to False.
        
        Returns:
            action (Action): action shape depends on the constructor arguments
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_prob(self, action: Action) -> torch.Tensor:
        """
        Returns the log of the probability mass/density function according to the `action`.
        
        Returns:
            log_prob (Tensor): `(*batch_shape, num_branches)`
        """
        raise NotImplementedError
    
    def joint_log_prob(self, action: Action) -> torch.Tensor:
        """
        Returns the joint log of the probability mass/density function according to the `action`.
        
        Returns:
            joint_log_prob (Tensor): `(*batch_shape, 1)`
        """
        return self.log_prob(action).sum(dim=-1, keepdim=True)
    
    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """
        Returns the entropy of this distribution. 
        
        Returns:
            entropy (Tensor): `(*batch_shape, num_branches)`
        """
        raise NotImplementedError
    
    def joint_entropy(self) -> torch.Tensor:
        """
        Returns the joint entropy of this distribution. 
        
        Returns:
            joint_entropy (Tensor): `(*batch_shape, 1)`
        """
        return self.entropy().sum(dim=-1, keepdim=True)

class CategoricalDist(PolicyDist):
    """
    Categorical policy distribution for the discrete action type. 
    It's parameterized by either `probs` or `logits` (but not both). 
    
    The shapes of parameters are `(*batch_shape, num_discrete_actions)` * `num_discrete_branches`.
    """
    def __init__(self, probs: tuple[torch.Tensor, ...] | None = None, logits: tuple[torch.Tensor, ...] | None = None) -> None:
        if probs is None and logits is None:
            raise ValueError("either probs or logits must be specified.")

        if probs is not None:
            self._dist = tuple(D.Categorical(probs=prob) for prob in probs)
        else:
            self._dist = tuple(D.Categorical(logits=logit) for logit in logits) # type: ignore
        
    def sample(self, _: bool = False) -> Action:
        sampled_discrete_action = [dist.sample() for dist in self._dist]
        sampled_discrete_action = torch.stack(sampled_discrete_action, dim=-1)
        return Action(discrete_action=sampled_discrete_action)
    
    def log_prob(self, action: Action) -> torch.Tensor:
        action_log_prob = []
        for i, dist in enumerate(self._dist):
            action_log_prob.append(dist.log_prob(action.discrete_action[..., i]))
        return torch.stack(action_log_prob, dim=-1)
    
    def entropy(self) -> torch.Tensor:
        entropies = [dist.entropy() for dist in self._dist]
        return torch.stack(entropies, dim=-1)

class GaussianDist(PolicyDist):
    """
    Gaussian policy distribution for the continuous action type.
    It's parameterized by `mean` and `std`.
    
    The shapes of both `mean` and `std` are `(*batch_shape, num_continuous_branches)`.
    `std` must be greater than 0.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self._dist = D.Normal(mean, std)
        
    def sample(self, reparam_trick: bool = False) -> Action:
        return Action(continuous_action=self._dist.rsample() if reparam_trick else self._dist.sample())
    
    def log_prob(self, action: Action) -> torch.Tensor:
        return self._dist.log_prob(action.continuous_action)
    
    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()
    
class CategoricalGaussianDist(PolicyDist):
    """
    Categorical-Gaussian policy distribution for the both discrete and continuous action types.
    
    See details in `CatetoricalDist` and `GaussianDist`.
    """
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        probs: tuple[torch.Tensor, ...] | None = None, 
        logits: tuple[torch.Tensor, ...] | None = None
    ) -> None:
        self._categorical_dist = CategoricalDist(probs, logits)
        self._gaussian_dist = GaussianDist(mean, std)
        
    def sample(self, reparam_trick: bool = False) -> Action:
        discrete_action = self._categorical_dist.sample()
        continuous_action = self._gaussian_dist.sample(reparam_trick)
        return Action(discrete_action.discrete_action, continuous_action.continuous_action)
    
    def log_prob(self, action: Action) -> torch.Tensor:
        discrete_log_prob = self._categorical_dist.log_prob(action)
        continuous_log_prob = self._gaussian_dist.log_prob(action)
        return torch.cat((discrete_log_prob, continuous_log_prob), dim=-1)
    
    def entropy(self) -> torch.Tensor:
        discrete_entropy = self._categorical_dist.entropy()
        continuous_entropy = self._gaussian_dist.entropy()
        return torch.cat((discrete_entropy, continuous_entropy), dim=-1)

class EpsilonGreedyDist(CategoricalDist):
    """
    Epsilon-greedy policy distribution for the discrete action type. 
    It's parameterized by `action_values` and `epsilon`. 
    
    The shape of `action_values` is `(*batch_shape, num_discrete_actions)` * `num_discrete_branches`.
    """
    def __init__(self, action_values: tuple[torch.Tensor, ...], epsilon: float) -> None:
        epsilon_greedy_probs = []
        for action_value in action_values:
            num_actions = action_value.shape[-1]
            
            # epsilon-greedy probabilities
            greedy_action_prob = 1.0 - epsilon + epsilon / num_actions
            non_greedy_action_prob = epsilon / num_actions
            
            # get greedy action
            greedy_action = action_value.argmax(dim=-1, keepdim=True)
            
            # set epsilon greedy probability distribution
            epsilon_greedy_prob = torch.full_like(action_value, fill_value=non_greedy_action_prob)
            epsilon_greedy_prob.scatter_(dim=-1, index=greedy_action, value=greedy_action_prob)
            epsilon_greedy_probs.append(epsilon_greedy_prob)
        
        super().__init__(probs=tuple(epsilon_greedy_probs))
    