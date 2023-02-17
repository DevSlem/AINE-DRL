from typing import Tuple
from abc import abstractmethod
from aine_drl.network import Network
from aine_drl.policy.policy_distribution import PolicyDistParam
import torch

class A2CSharedNetwork(Network[torch.Tensor]):
    """
    Advantage Actor Critic (A2C) shared network. 
    
    Note that since it uses the Actor-Critic architecure and the parameter sharing, 
    the encoding layer must be shared between Actor and Critic. 
    Therefore, single loss that is the sum of the actor and critic losses will be input.
    
    Generic type `T` is `Tensor`.
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> Tuple[PolicyDistParam, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution parameter (pdparam) and state value.

        Args:
            obs (Tensor): observation batch

        Returns:
            pdparam (PolicyDistParam): policy distribution parameter
            state_value (Tensor): state value V(s)
            
        ## Input/Output Details
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs|`(batch_size, *obs_shape)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |pdparam|`*batch_shape` = `(batch_size,)`, details in `PolicyDistParam` docs|
        |state_value|`(batch_size, 1)`|
        """
        raise NotImplementedError
