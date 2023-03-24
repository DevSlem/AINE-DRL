from abc import abstractmethod

from aine_drl.exp import Observation
from aine_drl.net import Network
from aine_drl.policy.policy import PolicyDistParam


class REINFORCENetwork(Network):
    """
    REINFORCE policy network.    
    """
    @abstractmethod
    def forward(self, obs: Observation) -> PolicyDistParam:
        """
        ## Summary
        
        Feed forward method to compute policy distribution parameter (pdparam).

        Args:
            obs (Observation): observation batch

        Returns:
            pdparam (PolicyDistParam): policy distribution parameter
            
        ## Input/Output Details
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs|`*batch_shape` = `(batch_size,)` details in `Observation` docs|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |pdparam|`*batch_shape` = `(batch_size,)`, details in `PolicyDistParam` docs|
        """
        raise NotImplementedError
