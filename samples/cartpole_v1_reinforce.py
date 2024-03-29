import sys

sys.path.append(".")

import torch.nn as nn
import torch.optim as optim

import aine_drl
import aine_drl.agent as agent
from aine_drl.factory import (AgentFactory, AINEInferenceFactory,
                              AINETrainFactory)
from aine_drl.policy import CategoricalPolicy


class CartPoleREINFORCENet(nn.Module, agent.REINFORCENetwork):    
    def __init__(self, obs_features, num_actions) -> None:
        super().__init__()
        
        # policy layer
        self.policy_net = nn.Sequential(
            nn.Linear(obs_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            CategoricalPolicy(64, num_actions)
        )
        
    def model(self) -> nn.Module:
        return self.policy_net
    
    def forward(self, obs: aine_drl.Observation) -> aine_drl.PolicyDist:
        return self.policy_net(obs.items[0])
    
class REINFORCEFactory(AgentFactory):
    def make(self, env: aine_drl.Env, config_dict: dict) -> agent.Agent:
        config = agent.REINFORCEConfig(**config_dict)
        
        network = CartPoleREINFORCENet(
            obs_features=env.obs_spaces[0][0],
            num_actions=env.action_space.discrete[0]
        )
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=0.001
        )).enable_grad_clip(network.parameters(), max_norm=5.0)
        
        return agent.REINFORCE(
            config,
            network,
            trainer,
        )
    
if __name__ == "__main__": 
    config_path = "config/samples/cartpole_v1_reinforce.yaml"
    
    AINETrainFactory.from_yaml(config_path) \
        .make_env() \
        .make_agent(REINFORCEFactory()) \
        .ready() \
        .train() \
        .close()
        
    AINEInferenceFactory.from_yaml(config_path) \
        .make_env() \
        .make_agent(REINFORCEFactory()) \
        .ready() \
        .inference() \
        .close()