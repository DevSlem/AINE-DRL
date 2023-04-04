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
from aine_drl.env import GymEnv, GymRenderableEnv
from aine_drl.factory import (AgentFactory, AINEInferenceFactory,
                              AINETrainFactory)
from aine_drl.policy import CategoricalPolicy

LEARNING_RATE = 3e-4
GRAD_CLIP_MAX_NORM = 5.0

class CartPoleNoVel(ObservationWrapper):
    def __init__(self, render_mode = None):
        env = gym.make("CartPole-v1", render_mode=render_mode)
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
        
        self.recurrent_layer_in_features = 64
        self.hiddeen_features = 128
        self.num_recurrent_layers = 1
        
        # encoding linear layer for feature extraction
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_features, self.recurrent_layer_in_features),
            nn.ReLU(),
            # nn.Linear(128, self.recurrent_layer_in_features),
            # nn.ReLU()
        )
        
        # recurrent layer for memory ability
        self.recurrent_layer = nn.LSTM(
            self.recurrent_layer_in_features, 
            self.hiddeen_features,
            num_layers=self.num_recurrent_layers,
            batch_first=True
        )
        
        # actor-critic layer
        self.actor = CategoricalPolicy(self.hiddeen_features, num_actions)
        self.critic = nn.Linear(self.hiddeen_features, 1)
        
    def hidden_state_shape(self) -> tuple[int, int]:
        return (self.num_recurrent_layers, self.hiddeen_features * 2)
        
    def model(self) -> nn.Module:
        return self
        
    def forward(self, obs_seq: aine_drl.Observation, hidden_state: torch.Tensor) -> tuple[aine_drl.PolicyDist, torch.Tensor, torch.Tensor]:
        vector_obs_seq = obs_seq.items[0]        
        # feed forward to encoding linear layer
        encoding_seq = self.encoding_layer(vector_obs_seq)
        
        # feed forward to recurrent layer
        h, c = net.unwrap_lstm_hidden_state(hidden_state)
        encoding_seq, (h_n, c_n) = self.recurrent_layer(encoding_seq, (h, c))
        next_seq_hidden_state = net.wrap_lstm_hidden_state(h_n, c_n)
        
        # feed forward to actor-critic layer
        policy_dist_seq = self.actor(encoding_seq)
        state_value_seq = self.critic(encoding_seq)
        
        return policy_dist_seq, state_value_seq, next_seq_hidden_state
    
class CartPoleNoVelNaivePPO(nn.Module, agent.PPOSharedNetwork):
    def __init__(self, obs_features, num_actions) -> None:
        super().__init__()
        
        self.hidden_features = 64
        
        # encoding layer for feature extraction
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_features, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_features),
            nn.ReLU(),
        )
        
        # actor-critic layer
        self.actor = CategoricalPolicy(self.hidden_features, num_actions)
        self.critic = nn.Linear(self.hidden_features, 1)
        
    def model(self) -> nn.Module:
        return self
    
    def forward(self, obs: aine_drl.Observation) -> tuple[aine_drl.PolicyDist, torch.Tensor]:
        vector_obs = obs.items[0]
        encoding = self.encoding_layer(vector_obs)
        policy_dist = self.actor(encoding)
        state_value = self.critic(encoding)
        return policy_dist, state_value
    
class RecurrentPPOFactory(AgentFactory):
    def make(self, env: aine_drl.Env, config_dict: dict) -> agent.Agent:
        config = agent.RecurrentPPOConfig(**config_dict)
        
        network = CartPoleNoVelRecurrentPPONet(
            obs_features=env.obs_spaces[0][0],
            num_actions=env.action_space.discrete[0]
        )
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=LEARNING_RATE
        )).enable_grad_clip(network.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            
        return agent.RecurrentPPO(
            config,
            network,
            trainer,
            env.num_envs
        )
        
class NaivePPOFactory(AgentFactory):
    def make(self, env: aine_drl.Env, config_dict: dict) -> agent.Agent:
        config = agent.PPOConfig(**config_dict)
        
        network = CartPoleNoVelNaivePPO(
            obs_features=env.obs_spaces[0][0],
            num_actions=env.action_space.discrete[0]
        )
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=LEARNING_RATE
        )).enable_grad_clip(network.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                
        return agent.PPO(
            config,
            network,
            trainer,
            env.num_envs
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inference", action="store_true", help="inference mode")
    parser.add_argument("-a", "--agent", type=str, default="recurrent_ppo", help="agent type (recurrent_ppo, naive_ppo)")
    args = parser.parse_args()
    is_inference = args.inference
    agent_type = args.agent
    
    recurrent_ppo_config_path = "config/experiments/cartpole_v1_no_velocity_recurrent_ppo.yaml"
    naive_ppo_config_path = "config/experiments/cartpole_v1_no_velocity_naive_ppo.yaml"
    
    if agent_type == "recurrent_ppo":
        config_path = recurrent_ppo_config_path
        agent_factory = RecurrentPPOFactory()
    elif agent_type == "naive_ppo":
        config_path = naive_ppo_config_path
        agent_factory = NaivePPOFactory()
    else:
        raise ValueError("invalid agent type")
    
    if not is_inference:
        aine_factory =  AINETrainFactory.from_yaml(config_path)
        
        env = GymEnv(gym.vector.AsyncVectorEnv([
            lambda: CartPoleNoVel() for _ in range(aine_factory.num_envs)
        ]), seed=aine_factory.seed)
        
        aine_factory.set_env(env) \
            .make_agent(agent_factory) \
            .ready() \
            .train() \
            .close()
    else:
        aine_factory = AINEInferenceFactory.from_yaml(config_path)
        
        env = GymRenderableEnv(CartPoleNoVel(render_mode="rgb_array"), seed=aine_factory.seed)
        
        aine_factory.set_env(env) \
            .make_agent(agent_factory) \
            .ready() \
            .inference() \
            .close()