import sys

sys.path.append(".")

import gym.vector
import aine_drl
import aine_drl.util as util
from aine_drl.training import GymTraining

import torch
import torch.nn as nn
import torch.optim as optim

class ObsEncodingLayer(nn.Module):
    def __init__(self, obs_shape, out_features) -> None:
        super().__init__()
        
        self.obs_shape = obs_shape
        self.out_features = out_features
        
        self.layers = nn.Sequential(
            nn.Linear(obs_shape, 256),
            nn.ReLU(),
            nn.Linear(256, out_features),
            nn.ReLU()
        )
        
    def forward(self, states):
        return self.layers(states)
    
class PolicyLayer(nn.Module):
    def __init__(self, in_features, continous_continous_action_count) -> None:
        super().__init__()
        
        self.in_features = in_features
        self.continous_continous_action_count = continous_continous_action_count
        
        self.layers = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, continous_continous_action_count * 2)
        )
        
    def forward(self, states):
        out = self.layers(states)
        out = torch.reshape(out, (-1, self.continous_continous_action_count, 2))
        torch.abs_(out[..., 1])
        return out
    
class ValueLayer(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        
        self.in_features = in_features
        
        self.layers = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, states):
        return self.layers(states)
    
class PolicyNet(nn.Module):
    def __init__(self, obs_encoding_layer, policy_layer) -> None:
        super().__init__()
        
        self.layers = nn.Sequential(
            obs_encoding_layer,
            policy_layer
        )
        
    def forward(self, states):
        return self.layers(states)
    
class ValueNet(nn.Module):
    def __init__(self, obs_encoding_layer, value_layer) -> None:
        super().__init__()
        
        self.layers = nn.Sequential(
            obs_encoding_layer,
            value_layer
        )
        
    def forward(self, states):
        return self.layers(states)
    
def main():
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    # training setting
    total_time_steps = 6000000
    summary_freq = 50000
    epoch = 15
    
    # create gym env
    num_envs = 32
    env = gym.vector.make("BipedalWalker-v3", num_envs=num_envs, new_step_api=True)
    obs_shape = env.single_observation_space.shape[0]
    continous_action_count = env.single_action_space.shape[0]
    
    # neural network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_features = 512
    obs_encoding_layer = ObsEncodingLayer(obs_shape, hidden_features)
    policy_layer = PolicyLayer(hidden_features, continous_action_count)
    value_layer = ValueLayer(hidden_features)
    policy_net = PolicyNet(obs_encoding_layer, policy_layer).to(device=device)
    value_net = ValueNet(obs_encoding_layer, value_layer).to(device=device)
    
    # optimizer
    params = list(obs_encoding_layer.parameters()) + list(policy_layer.parameters()) + list(value_layer.parameters())
    optimizer = optim.Adam(params, lr=3e-4)
    
    # PPO agent
    net_spec = aine_drl.ActorCriticSharedNetSpec(
        policy_net,
        value_net,
        optimizer,
        value_loss_coef=0.5,
        grad_clip_max_norm=5.0 # gradient clipping really really powerful!!!
    )
    categorical_policy = aine_drl.NormalPolicy()
    on_policy_trajectory = aine_drl.OnPolicyTrajectory(16, num_envs)
    ppo = aine_drl.PPO(
        net_spec,
        categorical_policy,
        on_policy_trajectory,
        aine_drl.Clock(num_envs),
        gamma=0.99,
        lam=0.95,
        epsilon_clip=0.2,
        entropy_coef=0.001,
        epoch=epoch,
        summary_freq=summary_freq
    )
    
    # start training
    gym_training = GymTraining(ppo, env, seed=seed, env_id="BipedalWalker-v3_PPO", auto_retrain=True)
    gym_training.run_train(total_time_steps)
    
if __name__ == "__main__":
    main()
