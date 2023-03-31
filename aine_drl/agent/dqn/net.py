from abc import abstractmethod

import torch

from aine_drl.exp import Observation
from aine_drl.net import Network
from aine_drl.policy_dist import PolicyDist

ActionValue = tuple[torch.Tensor, ...]

class DoubleDQNNetwork(Network):
    """
    Double Deep Q Network.     
    """
    @abstractmethod
    def forward(
        self, 
        obs: Observation
    ) -> tuple[PolicyDist, ActionValue]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution according to discrete action values.
        
        Args:
            obs (Observation): observation batch
            
        Returns:
            policy_dist (PolicyDist): policy distribution
            action_value (ActionValue): discrete action values (tuple of Tensors)
            
        ## Example
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs|`*batch_shape` = `(batch_size,)` details in `Observation` docs|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |policy_dist|`*batch_shape` = `(batch_size,)`, details in `PolicyDist` docs|
        |action_value|`(batch_size, num_discrete_actions)` x `num_discrete_branches`|
        """
        raise NotImplementedError
