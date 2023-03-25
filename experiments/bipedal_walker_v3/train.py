import sys
sys.path.append(".")

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import aine_drl
import aine_drl.util as util
from aine_drl.training import GymTraining

LEARNING_RATE = 3e-4

class BipedalWalkerPPONet(aine_drl.PPOSharedNetwork):
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
        
        self.add_model("encoding_layer", self.encoding_layer)
        self.add_model("actor_layer", self.actor_layer)
        self.add_model("critic_layer", self.critic_layer)
        
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        
        self.ts = aine_drl.TrainStep(self.optimizer)
        self.ts.enable_grad_clip(self.parameters(), grad_clip_max_norm=5.0)
    
    def forward(self, obs: torch.Tensor) -> tuple[aine_drl.PolicyDistParam, torch.Tensor]:
        encoding = self.encoding_layer(obs)
        pdparam = self.actor_layer(encoding)
        state_value = self.critic_layer(encoding)
        return pdparam, state_value
    
    def train_step(self, loss: torch.Tensor, training_step: int):
        self.ts.train_step(loss)

def get_bipedal_env_info(gym_training: GymTraining):
    return gym_training.observation_space.shape[0], gym_training.action_space.shape[0]

def auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
def run_ppo(inference: bool | None = False):
    # AINE-DRL configuration manager
    aine_config = aine_drl.AINEFactory("config/experiments/bipedal_walker_v3_ppo.yaml")
    
    # make gym training instance
    gym_training = aine_config.make_gym_training()
    
    # create actor-critic shared network
    obs_shape, num_actions = get_bipedal_env_info(gym_training)
    network = BipedalWalkerPPONet(obs_shape, num_actions).to(device=auto_device())
    
    # create policy for continuous action type
    policy = aine_drl.GaussianPolicy()
    
    # make PPO agent
    ppo = aine_config.make_agent(network, policy)
    
    if not inference:
        gym_training.train(ppo)
    else:
        gym_training.inference(ppo, num_episodes=10, agent_save_file_dir="experiments/bipedal_walker_v3/BipedalWalker-v3_PPO/agent.pt")
    
    # close safely
    gym_training.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="training")
    mode = parser.parse_args().mode
    
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    if mode == "training":
        run_ppo()
    elif mode == "inference": 
        run_ppo(inference=True)
    else:
        raise ValueError(f"\'training\', \'inference\' are only supported modes but you've input {mode}.")