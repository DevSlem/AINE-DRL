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

class QValueNet(nn.Module):
    def __init__(self, obs_shape, action_count) -> None:
        super(QValueNet, self).__init__()
        
        self.obs_shape = obs_shape
        self.action_count = action_count
        
        self.layers = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_count)
        )
        
    def forward(self, states):
        return self.layers(states)

def main():
    seed = 0 # if you want to get the same results
    venv_mode = True
    
    util.seed(seed)
    total_training_step = 150000
    training_freq = 32
    
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QValueNet(obs_shape, action_count).to(device=device)
    target_net = QValueNet(obs_shape, action_count).to(device=device)
    optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    dqn_spec = aine_drl.DQNSpec(
        q_net,
        target_net,
        optimizer,
    )
    clock = aine_drl.Clock(num_envs)
    epsilon_greedy = aine_drl.EpsilonGreedyPolicy(aine_drl.LinearDecay(0.3, 0.01, 0, total_training_step * 0.75))
    exp_replay = aine_drl.ExperienceReplay(training_freq, 32, 1000, num_envs)
    dqn = aine_drl.DoubleDQN(
        dqn_spec,
        epsilon_greedy,
        exp_replay,
        clock,
        gamma=0.99,
        epoch=3,
        summary_freq=1000,
        update_freq=128
    )
    gym_training = GymTraining(dqn, env, seed=seed)
    gym_training.run_train(total_training_step)
    
if __name__ == '__main__':
    main()
    