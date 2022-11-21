import sys
sys.path.append(".")

from typing import Optional, Tuple

import aine_drl
import aine_drl.util as util
from aine_drl.training import GymTraining

import torch
import torch.nn as nn
import torch.optim as optim

class CartPoleRecurrentActorCriticNet(aine_drl.RecurrentActorCriticSharedNetwork):
    
    def __init__(self, obs_shape, discrete_action_count) -> None:
        super().__init__()
        
        self.hidden_feature = 64
        self.lstm_in_feature = 64
        self._obs_shape = obs_shape
        
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_shape, self.lstm_in_feature),
            nn.ReLU()
        )
        
        self.lstm_layer = nn.LSTM(self.lstm_in_feature, self.hidden_feature, batch_first=True)
        
        self.actor_layer = aine_drl.DiscreteActionLayer(self.hidden_feature, discrete_action_count)
        self.critic_layer = nn.Linear(self.hidden_feature, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
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
    
    @property
    def obs_shape(self) -> torch.Size:
        return torch.Size((self._obs_shape,))
    
def main():
    seed = 0 # if you want to get the same results
    util.seed(seed)
    
    config_manager = aine_drl.util.ConfigManager("config/cartpole_v1_recurrent_ppo.yaml")
    gym_training = GymTraining.make(config_manager.env_config, config_manager.env_id)
    
    if gym_training.is_vector_env:
        obs_shape = gym_training.gym_env.single_observation_space.shape[0]
        action_count = gym_training.gym_env.single_action_space.n
    else:
        obs_shape = gym_training.gym_env.observation_space.shape[0]
        action_count = gym_training.gym_env.action_space.n
    
    device = None #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = CartPoleRecurrentActorCriticNet(obs_shape, action_count).to(device=device)
    policy = aine_drl.CategoricalPolicy()
    config = aine_drl.RecurrentPPOConfig(
        training_freq=64,
        epoch=5,
        sequence_length=8,
        num_sequence_batch=3,
        grad_clip_max_norm=5.0
    )
    recurrent_ppo = aine_drl.RecurrentPPO(
        config,
        network,
        policy,
        config_manager.env_config["num_envs"]
    )
    gym_training.train(recurrent_ppo)
    gym_training.close()
    
if __name__ == "__main__":
    main()
