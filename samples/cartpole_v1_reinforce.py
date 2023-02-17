import sys
sys.path.append(".")

import aine_drl
import aine_drl.util as util
import torch
import torch.nn as nn
import torch.optim as optim

class CartPoleREINFORCENet(aine_drl.REINFORCENetwork):    
    def __init__(self, obs_shape, discrete_action_count) -> None:
        super().__init__()
        
        # policy layer
        self.policy_net = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            aine_drl.DiscreteActionLayer(64, discrete_action_count)
        )
        
        # add models
        self.add_model("policy_net", self.policy_net)
        
        # optimizer for this network
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        self.ts = aine_drl.TrainStep(self.optimizer)
        self.ts.enable_grad_clip(self.parameters(), grad_clip_max_norm=5.0)
    
    # override
    def forward(self, obs: torch.Tensor) -> aine_drl.PolicyDistParam:
        return self.policy_net(obs)
    
    # override
    def train_step(self, loss: torch.Tensor, training_step: int):
        self.ts.train_step(loss)
    
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
    network = CartPoleREINFORCENet(obs_shape, action_count).to(device)
    
    # create policy for discrete action type
    policy = aine_drl.CategoricalPolicy()
    
    # make REINFORCE agent
    reinforce = aine_config.make_agent(network, policy)
    
    # training start!
    gym_training.train(reinforce)
    
    # training close safely
    gym_training.close()
