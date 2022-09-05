import torch
from torch.distributions import Categorical

class EpsilonGreedy(Categorical):
    """
    Epsilon greedy distribution. 
    The greater epsilon value, the more random the action is selected. 
    """
    
    def __init__(self, q_values: torch.Tensor, epsilon: float, validate_args=None):
        """
        Args:
            q_values (torch.Tensor): action values
            epsilon (float): [0, 1]
        """
        assert epsilon >= 0 and epsilon <= 1
        assert q_values.dim() >= 1 and q_values.dim() <= 2
        
        self._epsilon = epsilon
        
        num_action = q_values.shape[-1]
        # epsilon-greedy probabilities
        greedy_action_prob = 1.0 - epsilon + epsilon / num_action
        non_greedy_action_prob = epsilon / num_action
        # get greedy action
        greedy_action = q_values.argmax(-1)
        # set epsilon greedy probability distribution
        action_idx = torch.arange(num_action)
        x, y = torch.meshgrid(greedy_action, action_idx)
        dist = torch.empty_like(q_values).unsqueeze_(0)
        dist[..., x == y] = greedy_action_prob # greedy action
        dist[..., x != y] = non_greedy_action_prob # non-greedy action
        dist.squeeze_(0)
        
        super().__init__(probs=dist, validate_args=validate_args)
    
    @property
    def epsilon(self):
        return self._epsilon
