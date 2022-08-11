from lib2to3.pytree import Node
from aine_drl.policy import Policy
from aine_drl.drl_util import EpsilonGreedy
from aine_drl.util import aine_api, Decay, NoDecay
import torch
from torch.distributions import Distribution

class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float, epsilon_decay: Decay = None) -> None:
        assert epsilon >= 0 and epsilon <= 1
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else NoDecay(self.epsilon)
    
    @aine_api
    def get_policy_distribution(self, pdparam: torch.Tensor) -> Distribution:
        return EpsilonGreedy(pdparam, self.epsilon)
    
    @aine_api
    def update_hyperparams(self, time_step: int):
        self.epsilon = self.epsilon_decay(time_step)
        