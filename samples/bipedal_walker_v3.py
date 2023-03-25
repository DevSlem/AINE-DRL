import sys

sys.path.append(".")

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import aine_drl
from aine_drl.factory import (AgentFactory, AINEInferenceFactory,
                              AINETrainFactory)
from aine_drl.train import Env

LEARNING_RATE = 3e-4

class BipedalWalkerPPONet(nn.Module, aine_drl.PPOSharedNetwork):
    def __init__(self, obs_features, num_actions) -> None:
        super().__init__()
        
        self.hidden_feature = 256
        
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_features, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.hidden_feature)
        )
        
        self.actor_layer = aine_drl.GaussianLayer(self.hidden_feature, num_actions)
        self.critic_layer = nn.Linear(self.hidden_feature, 1)
    
    def model(self) -> nn.Module:
        return self
    
    def forward(self, obs: aine_drl.Observation) -> tuple[aine_drl.PolicyDistParam, torch.Tensor]:
        encoding = self.encoding_layer(obs.items[0])
        pdparam = self.actor_layer(encoding)
        state_value = self.critic_layer(encoding)
        return pdparam, state_value
        
class PPOFactory(AgentFactory):
    def make(self, env: Env, config_dict: dict) -> aine_drl.Agent:
        config = aine_drl.PPOConfig(**config_dict)
        
        network = BipedalWalkerPPONet(
            obs_features=env.obs_shape[0],
            num_actions=env.action_spec.num_continuous_actions
        ).to(device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=LEARNING_RATE
        )).enable_grad_clip(network.parameters(), max_norm=5.0)
        
        policy = aine_drl.GaussianPolicy()
        
        return aine_drl.PPO(
            config,
            network,
            trainer,
            policy,
            env.num_envs
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inference", action="store_true", help="inference mode")
    is_inference = parser.parse_args().inference
    
    ppo_config_path = "config/samples/bipedal_walker_v3_ppo.yaml"
    
    if not is_inference:
        AINETrainFactory.from_yaml(ppo_config_path) \
            .make_env() \
            .make_agent(PPOFactory()) \
            .ready() \
            .train() \
            .close()
    else:
        AINEInferenceFactory.from_yaml(ppo_config_path) \
            .make_env() \
            .make_agent(PPOFactory()) \
            .ready() \
            .inference() \
            .close()