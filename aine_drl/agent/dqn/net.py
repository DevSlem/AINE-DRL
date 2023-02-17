from typing import Tuple
from abc import abstractmethod
from aine_drl.network import Network
from aine_drl.policy.policy_distribution import PolicyDistParam
import torch

class DoubleDQNNetwork(Network[torch.Tensor]):
    """
    Double Deep Q Network. 
    
    Note that since it allows only discrete action type, `PolicyDistParam.discrete_pdparams` is only considered.
    
    Generic type `T` is `Tensor`.
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> PolicyDistParam:
        """
        ## Summary
        
        Feed forward method to compute policy distribution parameter (pdparam) consisting of discrete action values.
        
        Args:
            obs (Tensor): observation batch
            
        Returns:
            pdparam (PolicyDistParam): policy distribution parameter consisting of discrete action values Q(s, a)
            
        ## Example
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs|`(batch_size, *obs_shape)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |pdparam|`*batch_shape` = `(batch_size,)` and `*discrete_pdparam_shape` = `(num_discrete_actions,)`, details in `PolicyDistParam` docs|
        """
        raise NotImplementedError
