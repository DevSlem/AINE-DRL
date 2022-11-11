import sys
sys.path.append(".")

from typing import Optional, Tuple

import aine_drl
import aine_drl.util as util
from aine_drl.training import GymTraining

import torch
import torch.nn as nn
import torch.optim as optim

class CartPoleQValueNet(aine_drl.QValueNetwork):
    
    def __init__(self, obs_shape, discrete_action_count) -> None:
        super().__init__()
        
        self.q_value_layer = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            aine_drl.DiscreteActionLayer(64, discrete_action_count)
        )
                
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, obs: torch.Tensor) -> aine_drl.PolicyDistributionParameter:
        return self.q_value_layer(obs)
    
    def train_step(self, loss: torch.Tensor, grad_clip_max_norm: Optional[float], training_step: int):
        self.basic_train_step(loss, self.optimizer, grad_clip_max_norm)
    
def main():
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    config_manager = aine_drl.util.ConfigManager("config/cartpole_v1_double_dqn.yaml")
    gym_training = GymTraining.make(config_manager.env_config, config_manager.env_id)
    
    if gym_training.is_vector_env:
        obs_shape = gym_training.gym_env.single_observation_space.shape[0]
        action_count = gym_training.gym_env.single_action_space.n
    else:
        obs_shape = gym_training.gym_env.observation_space.shape[0]
        action_count = gym_training.gym_env.action_space.n
    
    device = None #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = CartPoleQValueNet(obs_shape, action_count).to(device=device)
    policy = aine_drl.EpsilonGreedyPolicy(aine_drl.drl_util.LinearDecay(0.3, 0.01, 1000, 120000))
    double_dqn = aine_drl.DoubleDQN.make(config_manager.env_config, network, policy)
    gym_training.train(double_dqn)
    gym_training.close()
    
if __name__ == "__main__":
    main()
