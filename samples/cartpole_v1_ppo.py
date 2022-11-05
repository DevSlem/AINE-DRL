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
    venv_mode = True
    
    util.seed(seed)
    total_time_steps = 200000
    
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
    
    network = CartPoleActorCriticNet(obs_shape, action_count)
    policy = aine_drl.CategoricalPolicy()
    config = aine_drl.PPOConfig(
        training_freq=16,
        epoch=3,
        mini_batch_size=8,
        gamma=0.99,
        lam=0.95,
        epsilon_clip=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.001,
        grad_clip_max_norm=5.0
    )
    
    ppo = aine_drl.PPO(
        config,
        network,
        policy,
        num_envs
    )
    
    gym_training = GymTraining(
        ppo, 
        env, 
        seed=seed, 
        env_id="CartPole-v1_PPO", 
        auto_retrain=False, 
        inference_freq=10000
    )
    gym_training.train(total_time_steps)
    
if __name__ == "__main__":
    main()