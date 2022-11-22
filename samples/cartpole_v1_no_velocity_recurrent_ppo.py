import sys
sys.path.append(".")

from typing import List, Optional, Tuple, Union

import aine_drl
import aine_drl.util as util
from aine_drl.training import GymTraining

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gym
import gym.spaces
import gym.vector

class CartPoleNoVelEnv(gym.vector.VectorEnv):
    """CartPole with no velocity."""

    def __init__(self, num_envs: int):
        self.gym_env = gym.vector.make("CartPole-v1", num_envs=num_envs, new_step_api=True)
        self.obs_mask = np.array([1, 0, 1, 0], dtype=np.bool8)
        low = self.gym_env.single_observation_space.low[self.obs_mask]
        high = self.gym_env.single_observation_space.high[self.obs_mask]
        
        super().__init__(num_envs, gym.spaces.Box(low, high), self.gym_env.single_action_space, True)
        
    def masked_obs(self, full_obs):
        return full_obs[:, self.obs_mask]

    def reset(self, *, seed: Optional[Union[int, List[int]]] = None, return_info: bool = False, options: Optional[dict] = None):
        full_obs = self.gym_env.reset(seed=seed, return_info=return_info, options=options)
        return self.masked_obs(full_obs)
        
    def step(self, actions):
        full_obs, reward, terminated, truncated, info = self.gym_env.step(actions)
        return self.masked_obs(full_obs), reward, terminated, truncated, info        

class CartPoleNoVelRecurrentActorCriticNet(aine_drl.RecurrentActorCriticSharedNetwork):
    
    def __init__(self, obs_shape, discrete_action_count) -> None:
        super().__init__()
        
        self.lstm_in_feature = 128
        self.hidden_feature = 64
        self._obs_shape = obs_shape
        
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, self.lstm_in_feature),
            nn.ReLU()
        )
        
        self.lstm_layer = nn.LSTM(self.lstm_in_feature, self.hidden_feature, batch_first=True)
        
        self.actor_layer = aine_drl.DiscreteActionLayer(self.hidden_feature, discrete_action_count)
        self.critic_layer = nn.Linear(self.hidden_feature, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
    
    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[aine_drl.PolicyDistributionParameter, torch.Tensor, torch.Tensor]:
        # encoding layer
        # (batch_size, seq_len, *obs_shape) -> (batch_size * seq_len, *obs_shape)
        seq_len = obs.shape[1]
        flattend = obs.flatten(0, 1)
        encoding = self.encoding_layer(flattend)
        
        # lstm layer
        unpacked_hidden_state = self.unpack_lstm_hidden_state(hidden_state)
        # (batch_size * seq_len, *lstm_in_feature) -> (batch_size, seq_len, *lstm_in_feature)
        encoding = encoding.reshape(-1, seq_len, self.lstm_in_feature)
        encoding, unpacked_hidden_state = self.lstm_layer(encoding, unpacked_hidden_state)
        next_hidden_state = self.pack_lstm_hidden_state(unpacked_hidden_state)
        
        # actor-critic layer
        # (batch_size, seq_len, *hidden_feature) -> (batch_size * seq_len, *hidden_feature)
        encoding = encoding.flatten(0, 1)
        pdparam = self.actor_layer(encoding)
        v_pred = self.critic_layer(encoding)
        
        return pdparam, v_pred, next_hidden_state
    
    def train_step(self, loss: torch.Tensor, grad_clip_max_norm: Optional[float], training_step: int):
        self.basic_train_step(loss, self.optimizer, grad_clip_max_norm)
        
    def hidden_state_shape(self, batch_size: int) -> torch.Size:
        return torch.Size((1, batch_size, self.hidden_feature * 2))
    
class CartPoleNoVelActorCriticNet(aine_drl.ActorCriticSharedNetwork):
    
    def __init__(self, obs_shape, discrete_action_count) -> None:
        super().__init__()
        
        self.hidden_feature = 128
        
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_feature),
            nn.ReLU()
        )
        
        self.actor_layer = aine_drl.DiscreteActionLayer(self.hidden_feature, discrete_action_count)
        self.critic_layer = nn.Linear(self.hidden_feature, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, obs: torch.Tensor) -> Tuple[aine_drl.PolicyDistributionParameter, torch.Tensor]:
        encoding = self.encoding_layer(obs)
        pdparam = self.actor_layer(encoding)
        v_pred = self.critic_layer(encoding)
        
        return pdparam, v_pred
    
    def train_step(self, loss: torch.Tensor, grad_clip_max_norm: Optional[float], training_step: int):
        self.basic_train_step(loss, self.optimizer, grad_clip_max_norm)
        
def train_recurrent_ppo():
    config_manager = aine_drl.util.ConfigManager("config/cartpole_v1_no_velocity_recurrent_ppo.yaml")
    num_envs = config_manager.env_config["num_envs"]
    gym_env = CartPoleNoVelEnv(num_envs)
    gym_training = GymTraining.make(config_manager.env_config, config_manager.env_id, gym_env=gym_env)
    
    if gym_training.is_vector_env:
        obs_shape = gym_training.gym_env.single_observation_space.shape[0]
        action_count = gym_training.gym_env.single_action_space.n
    else:
        obs_shape = gym_training.gym_env.observation_space.shape[0]
        action_count = gym_training.gym_env.action_space.n
    
    device = None #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = CartPoleNoVelRecurrentActorCriticNet(obs_shape, action_count).to(device=device)
    policy = aine_drl.CategoricalPolicy()
    config = aine_drl.RecurrentPPOConfig(
        training_freq=256,
        epoch=4,
        sequence_length=8,
        num_sequences_per_step=16,
        grad_clip_max_norm=5.0
    )
    recurrent_ppo = aine_drl.RecurrentPPO(config, network, policy, num_envs)
    gym_training.train(recurrent_ppo)
    gym_training.close() 
    
def train_ppo():
    config_manager = aine_drl.util.ConfigManager("config/cartpole_v1_no_velocity_ppo.yaml")
    num_envs = config_manager.env_config["num_envs"]
    gym_env = CartPoleNoVelEnv(num_envs)
    gym_training = GymTraining.make(config_manager.env_config, config_manager.env_id, gym_env=gym_env)
    
    if gym_training.is_vector_env:
        obs_shape = gym_training.gym_env.single_observation_space.shape[0]
        action_count = gym_training.gym_env.single_action_space.n
    else:
        obs_shape = gym_training.gym_env.observation_space.shape[0]
        action_count = gym_training.gym_env.action_space.n
        
    device = None #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = CartPoleNoVelActorCriticNet(obs_shape, action_count).to(device=device)
    policy = aine_drl.CategoricalPolicy()
    ppo = aine_drl.PPO.make(config_manager.env_config, network, policy)
    gym_training.train(ppo)
    gym_training.close()
    
def main():
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    train_recurrent_ppo()
    train_ppo()
    
if __name__ == "__main__":
    main()
