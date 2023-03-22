from abc import abstractmethod

import torch
import torch.nn as nn

from aine_drl.net import Network
from aine_drl.policy.policy import PolicyDistParam


class DoubleDQNNetwork(Network):
    """
    Double Deep Q Network. 
    
    Note that since it allows only discrete action type, only `PolicyDistParam.discrete_pdparams` have to be considered.
    """
    
    @property
    @abstractmethod
    def update_net(self) -> nn.Module:
        raise NotImplementedError
    
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
