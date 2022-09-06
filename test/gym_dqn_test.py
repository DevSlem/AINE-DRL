import sys
sys.path.append(".")

import gym.vector
import aine_drl
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
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_count)
        )
        
    def forward(self, states):
        return self.layers(states)

def main():
    total_training_step = 100000
    training_freq = 32
    num_envs = 3
    env = gym.vector.make("CartPole-v1", num_envs=num_envs, new_step_api=True)
    obs_shape = env.single_observation_space.shape[0]
    action_count = env.single_action_space.n
    q_net = QValueNet(obs_shape, action_count)
    target_net = QValueNet(obs_shape, action_count)
    optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    dqn_spec = aine_drl.DQNSpec(
        q_net,
        target_net,
        optimizer
    )
    clock = aine_drl.Clock(num_envs)
    dqn = aine_drl.DQN(
        dqn_spec,
        clock,
        update_freq=256
    )
    epsilon_greedy = aine_drl.EpsilonGreedyPolicy(aine_drl.LinearDecay(0.3, 0.01, 0, total_training_step))
    exp_replay = aine_drl.ExperienceReplay(training_freq, 32, 1000, num_envs, 3)
    agnet = aine_drl.GymAgent(
        env,
        dqn,
        epsilon_greedy,
        exp_replay,
        clock,
        summary_freq=100
    )
    agnet.train(total_training_step)
    
if __name__ == '__main__':
    main()
    