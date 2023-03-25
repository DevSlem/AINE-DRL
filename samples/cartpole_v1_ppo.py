import sys

sys.path.append(".")

import torch
import torch.nn as nn
import torch.optim as optim

import aine_drl
import aine_drl.util as util
from aine_drl.train import Env


class CartPolePPONet(nn.Module, aine_drl.PPOSharedNetwork):    
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
        self.actor_layer = aine_drl.CategoricalLayer(self.hidden_feature, num_actions)
        self.critic_layer = nn.Linear(self.hidden_feature, 1)
    
    def model(self) -> nn.Module:
        return self
    
    # override
    def forward(self, obs: aine_drl.Observation) -> tuple[aine_drl.PolicyDistParam, torch.Tensor]:
        encoding = self.encoding_layer(obs.items[0])
        pdparam = self.actor_layer(encoding)
        state_value = self.critic_layer(encoding)
        return pdparam, state_value
    
class PPOFactory(aine_drl.AgentFactory):
    def make(self, env: Env, config_dict: dict) -> aine_drl.Agent:
        config = aine_drl.PPOConfig(**config_dict)
        network = CartPolePPONet(
            obs_features=env.obs_shape[0],
            num_actions=env.action_spec.num_discrete_actions[0]
        ).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=0.001
        )).enable_grad_clip(network.parameters(), max_norm=5.0)
        policy = aine_drl.CategoricalPolicy()
        return aine_drl.PPO(
            config,
            network,
            trainer,
            policy,
            env.num_envs
        )
    
if __name__ == "__main__":
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    config_path = "config/samples/cartpole_v1_ppo.yaml"
    
    aine_drl.AINETrainFactory \
        .from_yaml(config_path) \
        .make_env() \
        .make_agent(PPOFactory()) \
        .ready() \
        .train() \
        .close()
        
    aine_drl.AINEInferenceFactory \
        .from_yaml(config_path) \
        .make_env() \
        .make_agent(PPOFactory()) \
        .ready() \
        .inference() \
        .close()