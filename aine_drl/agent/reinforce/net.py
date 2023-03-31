from abc import abstractmethod

from aine_drl.exp import Observation
from aine_drl.net import Network
from aine_drl.policy_dist import PolicyDist


class REINFORCENetwork(Network):
    """
    REINFORCE policy network.    
    """
    @abstractmethod
    def forward(
        self, 
        obs: Observation
    ) -> PolicyDist:
        """
        ## Summary
        
        Feed forward method to compute policy distribution.

        Args:
            obs (Observation): observation batch

        Returns:
            policy_dist (PolicyDist): policy distribution
            
        ## Input/Output Details
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs|`*batch_shape` = `(batch_size,)` details in `Observation` docs|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |policy_dist|`*batch_shape` = `(batch_size,)`, details in `PolicyDist` docs|
        """
        raise NotImplementedError
