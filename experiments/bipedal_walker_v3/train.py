import sys
sys.path.append(".")

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import aine_drl
import aine_drl.util as util

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
        self.basic_train_step(loss, self.optimizer, grad_clip_max_norm)
        
def train_ppo():
    # AINE-DRL configuration manager
    aine_config = aine_drl.AINEConfig("config/experiments/bipedal_walker_v3_ppo.yaml")
    
    # make gym training instance
    gym_training = aine_config.make_gym_training()
    
    # create actor-critic shared network
    obs_shape = gym_training.observation_space.shape[0]
    num_continuous_actions = gym_training.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = BipedalWalkerActorCriticNet(obs_shape, num_continuous_actions).to(device=device)
    
    # create policy for continuous action type
    policy = aine_drl.GaussianPolicy()
    
    # make PPO agent
    ppo = aine_config.make_agent(network, policy)
    
    # training start!
    gym_training.train(ppo)
    
    # training close safely
    gym_training.close()
    
def train_sac():
    # will be added soon after SAC is implemented
    pass
    
if __name__ == "__main__":
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    train_ppo()
    train_sac()