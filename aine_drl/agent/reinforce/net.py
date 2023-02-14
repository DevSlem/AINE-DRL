from abc import abstractmethod
from aine_drl.network import Network
from aine_drl.policy.policy_distribution import PolicyDistParam
import torch

class REINFORCENetwork(Network[torch.Tensor]):
    """
    Simple REINFORCE policy network.
    Generic type `T` is `Tensor`.
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
