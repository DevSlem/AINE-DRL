import sys
sys.path.append(".")

from typing import Optional, Tuple

import aine_drl
import aine_drl.util as util
from aine_drl.training import GymTraining

import torch
import torch.nn as nn
import torch.optim as optim

class BipedalWalkerActorCriticNet(aine_drl.ActorCriticSharedNetwork):
    
    def __init__(self, obs_shape, num_continuous_actions) -> None:
        super().__init__()
        
        self.hidden_feature = 256
        
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.hidden_feature)
        )
        
        self.actor_layer = aine_drl.GaussianContinuousActionLayer(self.hidden_feature, num_continuous_actions)
        self.critic_layer = nn.Linear(self.hidden_feature, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
    
    def forward(self, obs: torch.Tensor) -> Tuple[aine_drl.PolicyDistributionParameter, torch.Tensor]:
        encoding = self.encoding_layer(obs)
        pdparam = self.actor_layer(encoding)
        v_pred = self.critic_layer(encoding)
        
        return pdparam, v_pred
    
    def train_step(self, loss: torch.Tensor, grad_clip_max_norm: Optional[float], training_step: int):
        util.train_step(loss, self.optimizer, grad_clip_max_norm=grad_clip_max_norm, epoch=training_step)
    
def main():
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    config_manager = aine_drl.util.ConfigManager("config/bipedal_walker_v3_ppo.yaml")
    gym_training = GymTraining.make(config_manager.env_config, config_manager.env_id)
    
    obs_shape = gym_training.gym_env.single_observation_space.shape[0]
    continuous_action_count = gym_training.gym_env.single_action_space.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = BipedalWalkerActorCriticNet(obs_shape, continuous_action_count).to(device=device)
    policy = aine_drl.GaussianPolicy()
    ppo = aine_drl.PPO.make(config_manager.env_config, network, policy)
    gym_training.train(ppo)
    gym_training.close()
    
if __name__ == "__main__":
    main()
