import sys

sys.path.append(".")

import argparse

import gym
import gym.spaces
import gym.vector
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.core import ObservationWrapper

import aine_drl
import aine_drl.agent as agent
import aine_drl.net as net
from aine_drl.factory import AgentFactory, AINETrainFactory
from aine_drl.train import Env, GymEnv


class CartPoleNoVel(ObservationWrapper):
    def __init__(self):
        env = gym.make("CartPole-v1")
        super().__init__(env)
        
        self._obs_mask = np.array([True, False, True, False])
        
        self.observation_space = gym.spaces.Box(
            env.observation_space.low[self._obs_mask], # type: ignore
            env.observation_space.high[self._obs_mask], # type: ignore
        )
        
    def observation(self, observation):
        return observation[self._obs_mask]
    
class CartPoleNoVelRecurrentPPONet(nn.Module, agent.RecurrentPPOSharedNetwork):
    def __init__(self, obs_features, num_actions) -> None:
        super().__init__()
        
        self.recurrent_layer_in_features = 128
        self.hiddeen_features = 64
        self.num_recurrent_layers = 1
        
        # encoding linear layer for feature extraction
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_features, 64),
            nn.ReLU(),
            nn.Linear(64, self.recurrent_layer_in_features),
            nn.ReLU()
        )
        
        # recurrent layer for memory ability
        self.recurrent_layer = nn.LSTM(
            self.recurrent_layer_in_features, 
            self.hiddeen_features,
            num_layers=self.num_recurrent_layers,
            batch_first=True
        )
        
        # actor-critic layer
        self.actor_layer = aine_drl.CategoricalLayer(self.hiddeen_features, num_actions)
        self.critic_layer = nn.Linear(self.hiddeen_features, 1)
        
    def hidden_state_shape(self) -> tuple[int, int]:
        return (self.num_recurrent_layers, self.hiddeen_features * 2)
        
    def model(self) -> nn.Module:
        return self
        
    def forward(self, obs_seq: aine_drl.Observation, hidden_state: torch.Tensor) -> tuple[aine_drl.PolicyDistParam, torch.Tensor, torch.Tensor]:
        vector_obs_seq = obs_seq.items[0]
        seq_batch_size, seq_len, _ = self.unpack_seq_shape(vector_obs_seq)
        
        # feed forward to encoding linear layer
        vector_obs = vector_obs_seq.reshape(seq_batch_size * seq_len, -1)
        encoding = self.encoding_layer(vector_obs)
        
        # feed forward to recurrent layer
        h, c = net.unwrap_lstm_hidden_state(hidden_state)
        encoding_seq = encoding.reshape(seq_batch_size, seq_len, -1)
        encoding_seq, (h_n, c_n) = self.recurrent_layer(encoding_seq, (h, c))
        next_seq_hidden_state = net.wrap_lstm_hidden_state(h_n, c_n)
        
        # feed forward to actor-critic layer
        # (seq_batch_size, seq_len, hiddeen_features) -> (seq_batch_size * seq_len, hiddeen_features)
        encoding = encoding_seq.reshape(seq_batch_size * seq_len, -1)
        pdparam = self.actor_layer(encoding)
        state_value = self.critic_layer(encoding)
        
        # (seq_batch_size * seq_len, hiddeen_features) -> (seq_batch_size, seq_len, hiddeen_features)
        pdparam_seq = pdparam.transform(lambda x: x.reshape(seq_batch_size, seq_len, *x.shape[1:]))
        state_value_seq = state_value.reshape(seq_batch_size, seq_len, -1)
        
        return pdparam_seq, state_value_seq, next_seq_hidden_state
        
    @staticmethod
    def pack_lstm_hidden_state(h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """`(D x num_layers, num_seq, H_out) x 2` -> `(D x num_layers, num_seq, H_out x 2)`"""
        return torch.cat((h, c), dim=2)
    
    @staticmethod
    def unpack_lstm_hidden_state(lstm_hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(D x num_layers, num_seq, H_out x 2)` -> `(D x num_layers, num_seq, H_out) x 2`"""
        lstm_hidden_state = lstm_hidden_state.split(lstm_hidden_state.shape[2] // 2, dim=2)  # type: ignore
        return (lstm_hidden_state[0].contiguous(), lstm_hidden_state[1].contiguous())
    
class RecurrentPPOFactory(AgentFactory):
    def make(self, env: Env, config_dict: dict) -> agent.Agent:
        config = agent.RecurrentPPOConfig(**config_dict)
        
        network = CartPoleNoVelRecurrentPPONet(
            obs_features=env.obs_shape[0],
            num_actions=env.action_spec.num_discrete_actions[0]
        ).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=3e-4
        )).enable_grad_clip(network.parameters(), max_norm=5.0)
        
        policy = aine_drl.CategoricalPolicy()
        
        return agent.RecurrentPPO(
            config,
            network,
            trainer,
            policy,
            env.num_envs
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inference", action="store_true", help="inference mode")
    is_inference = parser.parse_args().inference
    
    recurrent_ppo_config_path = "config/samples/cartpole_v1_no_velocity_recurrent_ppo.yaml"
    
    if not is_inference:
        aine_factory =  AINETrainFactory.from_yaml(recurrent_ppo_config_path)
        
        env = GymEnv(gym.vector.AsyncVectorEnv([
            lambda: CartPoleNoVel() for _ in range(aine_factory.num_envs)
        ]))
        
        aine_factory.set_env(env) \
            .make_agent(RecurrentPPOFactory()) \
            .ready() \
            .train() \
            .close()
    else:
        raise NotImplementedError