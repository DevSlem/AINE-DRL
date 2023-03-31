from abc import abstractmethod

import torch

from aine_drl.exp import Observation
from aine_drl.net import Network
from aine_drl.policy_dist import PolicyDist


class A2CSharedNetwork(Network):
    """
    Advantage Actor Critic (A2C) shared network. 
    
    Note that since it uses the Actor-Critic architecure and the parameter sharing, 
    the encoding layer must be shared between Actor and Critic.
    """
    @abstractmethod
    def forward(
        self, 
        obs: Observation
    ) -> tuple[PolicyDist, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution and state value.

        Args:
            obs (Observation): observation batch

        Returns:
            policy_dist (PolicyDist): policy distribution
            state_value (Tensor): state value V(s)
            
        ## Input/Output Details
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs|`*batch_shape` = `(batch_size,)` details in `Observation` docs|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |policy_dist|`*batch_shape` = `(batch_size,)`, details in `PolicyDist` docs|
        |state_value|`(batch_size, 1)`|
        """
        raise NotImplementedError
