import sys
from typing import Optional, Tuple

sys.path.append(".")

import gym
import gym.vector
import aine_drl
import aine_drl.util as util
from aine_drl.training import GymTraining

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

class CartPoleActorCriticNet(aine_drl.ActorCriticSharedNetwork):
    
    def __init__(self, obs_shape, discrete_action_count) -> None:
        super().__init__()
        
        self.hidden_feature = 64
        
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_feature),
            nn.ReLU()
        )
        
        self.actor_layer = nn.Linear(self.hidden_feature, discrete_action_count)
        self.critic_layer = nn.Linear(self.hidden_feature, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, obs: torch.Tensor) -> Tuple[aine_drl.PolicyDistributionParameter, torch.Tensor]:
        encoding = self.encoding_layer(obs)
        discrete_pdparam = self.actor_layer(encoding)
        v_pred = self.critic_layer(encoding)
        
        return aine_drl.PolicyDistributionParameter.create([discrete_pdparam], None), v_pred
    
    def train_step(self, loss: torch.Tensor, grad_clip_max_norm: Optional[float], training_step: int):
        util.train_step(loss, self.optimizer, grad_clip_max_norm=grad_clip_max_norm, epoch=training_step)
    
def main():
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    config_manager = aine_drl.util.ConfigManager("config/cartpole_v1_ppo.yaml")
    gym_training = GymTraining.make(config_manager.env_config, config_manager.env_id)
    
    if gym_training.is_vector_env:
        obs_shape = gym_training.gym_env.single_observation_space.shape[0]
        action_count = gym_training.gym_env.single_action_space.n
    else:
        obs_shape = gym_training.gym_env.observation_space.shape[0]
        action_count = gym_training.gym_env.action_space.n
    
    device = None #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = CartPoleActorCriticNet(obs_shape, action_count).to(device=device)
    policy = aine_drl.CategoricalPolicy()
    ppo = aine_drl.PPO.make(config_manager.env_config, network, policy)
    gym_training.train(ppo)
    
if __name__ == "__main__":
    main()
