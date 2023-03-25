import sys

sys.path.append(".")

import torch.nn as nn
import torch.optim as optim

import aine_drl
from aine_drl.factory import (AgentFactory, AINEInferenceFactory,
                              AINETrainFactory)
from aine_drl.train import Env


class CartPoleREINFORCENet(nn.Module, aine_drl.REINFORCENetwork):    
    def __init__(self, obs_features, num_actions) -> None:
        super().__init__()
        
        # policy layer
        self.policy_net = nn.Sequential(
            nn.Linear(obs_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            aine_drl.CategoricalLayer(64, num_actions)
        )
        
    def model(self) -> nn.Module:
        return self.policy_net
    
    def forward(self, obs: aine_drl.Observation) -> aine_drl.PolicyDistParam:
        return self.policy_net(obs.items[0])
    
class REINFORCEFactory(AgentFactory):
    def make(self, env: Env, config_dict: dict) -> aine_drl.Agent:
        config = aine_drl.REINFORCEConfig(**config_dict)
        
        network = CartPoleREINFORCENet(
            obs_features=env.obs_shape[0],
            num_actions=env.action_spec.num_discrete_actions[0]
        )
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=0.001
        )).enable_grad_clip(network.parameters(), max_norm=5.0)
        
        policy = aine_drl.CategoricalPolicy()
        
        return aine_drl.REINFORCE(
            config,
            network,
            trainer,
            policy,
        )
    
if __name__ == "__main__": 
    config_path = "config/samples/cartpole_v1_reinforce.yaml"
    
    AINETrainFactory \
        .from_yaml(config_path) \
        .make_env() \
        .make_agent(REINFORCEFactory()) \
        .ready() \
        .train() \
        .close()
        
    AINEInferenceFactory \
        .from_yaml(config_path) \
        .make_env() \
        .make_agent(REINFORCEFactory()) \
        .ready() \
        .inference() \
        .close()