import sys

sys.path.append(".")

import torch
import torch.nn as nn
import torch.optim as optim

import aine_drl
import aine_drl.agent as agent
from aine_drl.factory import AgentFactory, AINETrainFactory
from aine_drl.train import Env

from math import floor

class GridWorldPPONet(nn.Module, agent.PPOSharedNetwork):
    def __init__(self, obs_shape, num_actions) -> None:
        super().__init__()
        
        height = obs_shape[0]
        width = obs_shape[1]
        channel = obs_shape[2]
        
        conv_out_shape = self.conv_output_shape((height, width), 7)
        conv_out_shape = self.conv_output_shape(conv_out_shape, 3)
        hidden_features = conv_out_shape[0] * conv_out_shape[1] * 32
        
        self.encoding_layer = nn.Sequential(
            nn.Conv2d(channel, 16, 7),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.actor = aine_drl.CategoricalLayer(hidden_features, num_actions)
        self.critic = nn.Linear(hidden_features, 1)
        
    def model(self) -> nn.Module:
        return self
        
    def forward(self, obs: aine_drl.Observation) -> tuple[aine_drl.PolicyDistParam, torch.Tensor]:
        image_obs = obs.items[0].permute(0, 3, 1, 2)
        encoding = self.encoding_layer(image_obs)
        pdparam = self.actor(encoding)
        state_value = self.critic(encoding)
        return pdparam, state_value
    
    @staticmethod
    def conv_output_shape(
        h_w: tuple[int, int],
        kernel_size: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
    ):
        """
        Computes the height and width of the output of a convolution layer.
        """
        h = floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        w = floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        return h, w

class PPOFactory(AgentFactory):
    def make(self, env: Env, config_dict: dict) -> agent.Agent:
        config = agent.PPOConfig(**config_dict)
        
        network = GridWorldPPONet(
            obs_shape=env.obs_spaces[0],
            num_actions=env.action_space.discrete[0]
        ).to(device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=3e-4
        )).enable_grad_clip(network.parameters(), max_norm=5.0)
        
        policy = aine_drl.CategoricalPolicy()
        
        return agent.PPO(
            config,
            network,
            trainer,
            policy,
            env.num_envs
        )

if __name__ == "__main__":
    AINETrainFactory.from_yaml("config/samples/gridworld_ppo.yaml") \
        .make_env() \
        .make_agent(PPOFactory()) \
        .ready() \
        .train() \
        .close()