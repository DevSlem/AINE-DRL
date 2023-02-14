import sys
sys.path.append(".")

from typing import Optional, Tuple

import aine_drl
import aine_drl.util as util

import torch
import torch.nn as nn
import torch.optim as optim

class CartPoleA2CNet(aine_drl.A2CSharedNetwork):
    # A2C uses ActorCriticSharedNetwork.
    
    def __init__(self, obs_shape, discrete_action_count) -> None:
        super().__init__()
        
        self.hidden_feature = 64
    
        # encoding layer for shared network
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_feature),
            nn.ReLU()
        )
        
        # actor-critic layer
        self.actor_layer = aine_drl.DiscreteActionLayer(self.hidden_feature, discrete_action_count)
        self.critic_layer = nn.Linear(self.hidden_feature, 1)
        
        # optimizer for this network
        self.optimizer = optim.Adam(
            self.concat_model_params(self.encoding_layer, self.actor_layer, self.critic_layer), 
            lr=0.001
        )
        
        self.add_model("encoding_layer", self.encoding_layer)
        self.add_model("actor_layer", self.actor_layer)
        self.add_model("critic_layer", self.critic_layer)
    
    # override
    def forward(self, obs: torch.Tensor) -> Tuple[aine_drl.PolicyDistParam, torch.Tensor]:
        encoding = self.encoding_layer(obs)
        pdparam = self.actor_layer(encoding)
        state_value = self.critic_layer(encoding)
        return pdparam, state_value
    
    # override
    def train_step(self, loss: torch.Tensor, grad_clip_max_norm: Optional[float], training_step: int):
        self.simple_train_step(loss, self.optimizer, grad_clip_max_norm)
    
if __name__ == "__main__":
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    # AINE-DRL configuration manager
    aine_config = aine_drl.AINEConfig("config/samples/cartpole_v1_a2c.yaml")
    
    # make gym training instance
    gym_training = aine_config.make_gym_training()
    
    # create custom network
    obs_shape = gym_training.observation_space.shape[0]
    action_count = gym_training.action_space.n
    device = None #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = CartPoleA2CNet(obs_shape, action_count).to(device=device)
    
    # create policy for discrete action type
    policy = aine_drl.CategoricalPolicy()
    
    # make A2C agent
    a2c = aine_config.make_agent(network, policy)
    
    # training start!
    gym_training.train(a2c)
    
    # training close safely
    gym_training.close()
