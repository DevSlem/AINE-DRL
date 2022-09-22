import sys

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

class ObsEncoderLayer(nn.Module):
    def __init__(self, obs_shape, out_features) -> None:
        super().__init__()
        
        self.obs_shape = obs_shape
        self.out_features = out_features
        
        self.layers = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
            nn.ReLU()
        )
        
    def forward(self, states):
        return self.layers(states)

class PolicyNet(nn.Module):
    def __init__(self, obs_encoder_layer, policy_layer) -> None:
        super().__init__()
        
        self.layers = nn.Sequential(
            obs_encoder_layer,
            policy_layer
        )
        
    def forward(self, states):
        return self.layers(states)
    
class ValueNet(nn.Module):
    def __init__(self, obs_encoder_layer, value_layer) -> None:
        super().__init__()
        
        self.layers = nn.Sequential(
            obs_encoder_layer,
            value_layer
        )
        
    def forward(self, states):
        return self.layers(states)
    
def main():
    seed = 0 # if you want to get the same results
    venv_mode = True
    
    util.seed(seed)
    total_time_steps = 400000
    epoch = 3
    
    if venv_mode:
        num_envs = 3
        env = gym.vector.make("CartPole-v1", num_envs=num_envs, new_step_api=True)
        obs_shape = env.single_observation_space.shape[0]
        action_count = env.single_action_space.n
    else:
        num_envs = 1
        env = gym.make("CartPole-v1", new_step_api=True)
        obs_shape = env.observation_space.shape[0]
        action_count = env.action_space.n
    
    device = None #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_encoder_layer = ObsEncoderLayer(obs_shape, 64)
    policy_layer = nn.Sequential(
        nn.Linear(obs_encoder_layer.out_features, action_count)
    )
    value_layer = nn.Sequential(
        nn.Linear(obs_encoder_layer.out_features, 1)
    )
    
    policy_net = PolicyNet(obs_encoder_layer, policy_layer).to(device=device)
    value_net = ValueNet(obs_encoder_layer, value_layer).to(device=device)
    
    params = list(obs_encoder_layer.parameters()) + list(policy_layer.parameters()) + list(value_layer.parameters())
    optimizer = optim.Adam(params, lr=0.001)
    
    net_spec = aine_drl.ActorCriticSharedNetSpec(
        policy_net,
        value_net,
        optimizer,
        value_loss_coef=0.5
    )
    
    categorical_policy = aine_drl.CategoricalPolicy()
    on_policy_trajectory = aine_drl.OnPolicyTrajectory(12, num_envs)
    
    ppo = aine_drl.PPO(
        net_spec,
        categorical_policy,
        on_policy_trajectory,
        aine_drl.Clock(num_envs),
        gamma=0.99,
        lam=0.95,
        epsilon_clip=0.2,
        entropy_coef=0.001,
        epoch=epoch
    )
    
    gym_training = GymTraining(ppo, env, seed=seed, env_id="CartPole-v1_PPO", auto_retrain=False)
    gym_training.run_train(total_time_steps)
    
if __name__ == "__main__":
    main()