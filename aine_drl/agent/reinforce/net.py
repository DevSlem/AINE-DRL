from abc import ABC, abstractmethod

import torch

from aine_drl.net import Network
from aine_drl.policy.policy import PolicyDistParam


class REINFORCEOptim(ABC):
    @abstractmethod
    def step(self, loss: torch.Tensor, training_steps: int):
        raise NotImplementedError

class REINFORCENetwork(Network):
    """
    REINFORCE policy network.    
    """
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> PolicyDistParam:
        """
        ## Summary
        
        Feed forward method to compute policy distribution parameter (pdparam).

        Args:
            obs (Tensor): observation batch

        Returns:
            pdparam (PolicyDistParam): policy distribution parameter
            
        ## Input/Output Details
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs|`(batch_size, *obs_shape)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |pdparam|`*batch_shape` = `(batch_size,)`, details in `PolicyDistParam` docs|
        """
        raise NotImplementedError
