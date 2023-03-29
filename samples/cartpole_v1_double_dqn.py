import sys

sys.path.append(".")

import torch.nn as nn
import torch.optim as optim

import aine_drl
import aine_drl.agent as agent
from aine_drl.factory import (AgentFactory, AINEInferenceFactory,
                              AINETrainFactory)
from aine_drl.policy import EpsilonGreedyPolicy
from aine_drl.agent.dqn.net import ActionValue


class CartPoleDoubleDQNNet(nn.Module, agent.DoubleDQNNetwork):    
    def __init__(self, obs_features, num_actions) -> None:
        super().__init__()
        
        # Q value (or action value) layer
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.policy = EpsilonGreedyPolicy(64, num_actions, epsilon_decay=0.01)
        
    def model(self) -> nn.Module:
        return self
        
    def forward(self, obs: aine_drl.Observation) -> tuple[aine_drl.PolicyDist, ActionValue]:
        encoding = self.encoding_layer(obs.items[0])
        policy_dist = self.policy(encoding)
        return policy_dist, self.policy.pop_action_values()
    
class DoubleDQNFactory(AgentFactory):
    def make(self, env: aine_drl.Env, config_dict: dict) -> agent.Agent:
        config = agent.DoubleDQNConfig(**config_dict)
        
        network = CartPoleDoubleDQNNet(
            obs_features=env.obs_spaces[0][0],
            num_actions=env.action_space.discrete[0]
        )
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=0.001
        )).enable_grad_clip(network.parameters(), max_norm=5.0)
                
        return agent.DoubleDQN(
            config,
            network,
            trainer,
            env.num_envs
        )
    
if __name__ == "__main__":
    config_path = "config/samples/cartpole_v1_double_dqn.yaml"
    
    AINETrainFactory.from_yaml(config_path) \
        .make_env() \
        .make_agent(DoubleDQNFactory()) \
        .ready() \
        .train() \
        .close()
        
    AINEInferenceFactory.from_yaml(config_path) \
        .make_env() \
        .make_agent(DoubleDQNFactory()) \
        .ready() \
        .inference() \
        .close()