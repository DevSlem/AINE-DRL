import sys
sys.path.append(".")

from typing import Optional, Tuple

import aine_drl
import aine_drl.util as util
from aine_drl.training import GymTraining

import torch
import torch.nn as nn
import torch.optim as optim

class CartPolePolicyGradientcNet(aine_drl.PolicyGradientNetwork):
    
    def __init__(self, obs_shape, discrete_action_count) -> None:
        super().__init__()
        
        self.policy_layer = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            aine_drl.DiscreteActionLayer(64, discrete_action_count)
        )
                
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, obs: torch.Tensor) -> aine_drl.PolicyDistributionParameter:
        return self.policy_layer(obs)
    
    def train_step(self, loss: torch.Tensor, grad_clip_max_norm: Optional[float], training_step: int):
        util.train_step(loss, self.optimizer, grad_clip_max_norm=grad_clip_max_norm, epoch=training_step)
    
def main():
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    config_manager = aine_drl.util.ConfigManager("config/cartpole_v1_reinforce.yaml")
    gym_training = GymTraining.make(config_manager.env_config, config_manager.env_id)
    
    obs_shape = gym_training.gym_env.observation_space.shape[0]
    action_count = gym_training.gym_env.action_space.n
    
    device = None #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = CartPolePolicyGradientcNet(obs_shape, action_count).to(device=device)
    policy = aine_drl.CategoricalPolicy()
    reinforce = aine_drl.REINFORCE.make(config_manager.env_config, network, policy)
    gym_training.train(reinforce)
    gym_training.close()
    
if __name__ == "__main__":
    main()
