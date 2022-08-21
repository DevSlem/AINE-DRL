import importlib

class DRLAgentFactory:
    def __init__(self) -> None:
        self.drl_algorithm_module = "aine_drl.drl_algorithm"
        self.policy_module = "aine_drl.policy"
        self.trajectory_module = "aine_drl.trajectory"
        self.decay_module = "aine_drl.util"
        
    def create_drl_algorithm(self, config):
        pass
    
    def create_decay(self, config):
        pass
    
    def create_trajectory(self, config):
        pass