import sys

sys.path.append(".")

import torch
import torch.nn as nn
import torch.optim as optim

import aine_drl
import aine_drl.agent as agent
from aine_drl.factory import (AgentFactory, AINEInferenceFactory,
                              AINETrainFactory)
from aine_drl.policy import CategoricalPolicy


class CartPolePPONet(nn.Module, agent.PPOSharedNetwork):    
    def __init__(self, obs_features, num_actions) -> None:
        super().__init__()
        
        self.hidden_feature = 64
        
        # encoding layer for shared network
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_features, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_feature),
            nn.ReLU()
        )
        
        # actor-critic layer
        self.actor = CategoricalPolicy(self.hidden_feature, num_actions)
        self.critic = nn.Linear(self.hidden_feature, 1)
    
    def model(self) -> nn.Module:
        return self
    
    # override
    def forward(self, obs: aine_drl.Observation) -> tuple[aine_drl.PolicyDist, torch.Tensor]:
        encoding = self.encoding_layer(obs.items[0])
        policy_dist = self.actor(encoding)
        state_value = self.critic(encoding)
        return policy_dist, state_value
    
class PPOFactory(AgentFactory):
    def make(self, env: aine_drl.Env, config_dict: dict) -> agent.Agent:
        config = agent.PPOConfig(**config_dict)
        
        network = CartPolePPONet(
            obs_features=env.obs_spaces[0][0],
            num_actions=env.action_space.discrete[0]
        )
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=0.001
        )).enable_grad_clip(network.parameters(), max_norm=5.0)
                
        return agent.PPO(
            config,
            network,
            trainer,
            env.num_envs
        )
    
if __name__ == "__main__":  
    config_path = "config/samples/cartpole_v1_ppo.yaml"
    
    AINETrainFactory.from_yaml(config_path) \
        .make_env() \
        .make_agent(PPOFactory()) \
        .ready() \
        .train() \
        .close()
        
    AINEInferenceFactory.from_yaml(config_path) \
        .make_env() \
        .make_agent(PPOFactory()) \
        .ready() \
        .inference() \
        .close()