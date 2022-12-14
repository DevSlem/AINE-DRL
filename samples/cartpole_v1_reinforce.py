import sys
sys.path.append(".")

from typing import Optional
import aine_drl
import aine_drl.util as util
import torch
import torch.nn as nn
import torch.optim as optim

class CartPolePolicyGradientcNet(aine_drl.PolicyGradientNetwork):
    # REINFORCE uses PolicyGradientNetwork.
    
    def __init__(self, obs_shape, discrete_action_count) -> None:
        super().__init__()
        
        # policy layer
        self.policy_layer = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            aine_drl.DiscreteActionLayer(64, discrete_action_count)
        )
        
        # optimizer for this network
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    # override
    def forward(self, obs: torch.Tensor) -> aine_drl.PolicyDistributionParameter:
        return self.policy_layer(obs)
    
    # override
    def train_step(self, loss: torch.Tensor, grad_clip_max_norm: Optional[float], training_step: int):
        self.basic_train_step(loss, self.optimizer, grad_clip_max_norm)
    
if __name__ == "__main__":
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    # AINE-DRL configuration manager
    aine_config = aine_drl.AINEConfig("config/samples/cartpole_v1_reinforce.yaml")
    
    # make gym training instance
    gym_training = aine_config.make_gym_training()
    
    # create custom network
    obs_shape = gym_training.observation_space.shape[0]
    action_count = gym_training.action_space.n
    device = None #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = CartPolePolicyGradientcNet(obs_shape, action_count).to(device=device)
    
    # create policy for discrete action type
    policy = aine_drl.CategoricalPolicy()
    
    # make REINFORCE agent
    reinforce = aine_config.make_agent(network, policy)
    
    # training start!
    gym_training.train(reinforce)
    
    # training close safely
    gym_training.close()
