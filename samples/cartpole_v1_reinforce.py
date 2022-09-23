import sys

sys.path.append(".")

import gym
import gym.vector
import aine_drl
import aine_drl.util as util
from aine_drl.training import GymTraining

import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, obs_shape, discrete_action_count) -> None:
        super(PolicyNet, self).__init__()
        
        self.obs_shape = obs_shape
        self.discrete_action_count = discrete_action_count
        
        self.layers = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, discrete_action_count)
        )
        
    def forward(self, states):
        return self.layers(states)
    
def main():
    seed = 0 # if you want to get the same results
    venv_mode = False
    
    util.seed(seed)
    total_training_step = 300000
    
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
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = PolicyNet(obs_shape, action_count)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    net_spec = aine_drl.REINFORCENetSpec(
        policy_net,
        optimizer
    )
    categorical_policy = aine_drl.CategoricalPolicy()
    mc_trajectory = aine_drl.MonteCarloTrajectory(num_envs)
    reinforce = aine_drl.REINFORCE(
        net_spec,
        categorical_policy,
        mc_trajectory,
        aine_drl.Clock(num_envs),
        gamma=0.99
    )
    
    gym_training = GymTraining(
        reinforce, 
        env, 
        seed=seed, 
        env_id="CartPole-v1_REINFORCE", 
        auto_retrain=False, 
        render_freq=10000
    )
    gym_training.run_train(total_training_step)
    
if __name__ == "__main__":
    main()