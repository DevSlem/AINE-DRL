from typing import Dict, NamedTuple, Optional, Tuple
from aine_drl.agent import Agent
from aine_drl.experience import ActionTensor, Experience
from aine_drl.network import RecurrentActorCriticSharedNetwork
from aine_drl.policy.policy import Policy
from .ppo import PPO
from .ppo_trajectory import RecurrentPPOExperienceBatch, RecurrentPPOTrajectory
import aine_drl.drl_util as drl_util
import aine_drl.util as util
import torch
from torch.nn.utils.rnn import pad_sequence

class RecurrentPPOConfig(NamedTuple):
    """
    Recurrent PPO configuration.

    Args:
        training_freq (int): training frequency which is the number of time steps to gather experiences
        epoch (int): number of using total experiences to update parameters
        sequence_length (int): sequence length of recurrent network when training. trajectory is split by `sequence_length` unit. a value of `8` or greater are typically recommended.
        num_sequences_per_step (int): number of sequences per train step, which are selected randomly
        gamma (float, optional): discount factor. Defaults to 0.99.
        lam (float, optional): regularization parameter which controls the balanace of Generalized Advantage Estimation (GAE) between bias and variance. Defaults to 0.95.
        epsilon_clip (float, optional): clipping the probability ratio (pi_theta / pi_theta_old) to [1-eps, 1+eps]. Defaults to 0.2.
        value_loss_coef (float, optional): value loss multiplier. Defaults to 0.5.
        entropy_coef (float, optional): entropy multiplier. Defaults to 0.001.
        grad_clip_max_norm (float | None, optional): gradient clipping maximum value. Defaults to no gradient clipping.
    """
    training_freq: int
    epoch: int
    sequence_length: int
    num_sequences_per_step: int
    gamma: float = 0.99
    lam: float = 0.95
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    grad_clip_max_norm: Optional[float] = None
    

class RecurrentPPO(Agent):
    """
    Recurrent Proximal Policy Optimization (PPO) using RNN. See details in https://arxiv.org/abs/1707.06347.

    Args:
        config (RecurrentPPOConfig): Recurrent PPO configuration
        network (RecurrentActorCriticNetwork): recurrent actor critic network (e.g., LSTM, GRU)
        policy (Policy): policy
        num_envs (int): number of environments
    """
    def __init__(self, 
                 config: RecurrentPPOConfig,
                 network: RecurrentActorCriticSharedNetwork,
                 policy: Policy,
                 num_envs: int) -> None:        
        if not isinstance(network, RecurrentActorCriticSharedNetwork):
            raise ValueError("The network type must be RecurrentActorCriticSharedNetwork.")
        
        super().__init__(network, policy, num_envs)
        
        self.config = config
        self.network = network
        self.trajectory = RecurrentPPOTrajectory(self.config.training_freq)
        
        self.current_action_log_prob = None
        self.v_pred = None
        self.current_hidden_state = torch.zeros(network.hidden_state_shape(self.num_envs))
        self.next_hidden_state = torch.zeros(network.hidden_state_shape(self.num_envs))
        self.prev_terminated = torch.zeros(self.num_envs, 1)
        
        self.actor_average_loss = util.IncrementalAverage()
        self.critic_average_loss = util.IncrementalAverage()
        
    # @staticmethod
    # def make(env_config: dict,
    #          network: RecurrentActorCriticNetwork,
    #          policy: Policy):
    #     """
    #     ## Summary
        
    #     Helps to make PPO agent.

    #     Args:
    #         env_config (dict): environment configuration which inlcudes `num_envs`, `PPO`
    #         network (ActorCriticSharedNetwork): standard actor critic network
    #         policy (Policy): policy

    #     Returns:
    #         PPO: `PPO` instance
            
    #     ## Example
        
    #     `env_config` dictionary format::
        
    #         {'num_envs': 3,
    #          'PPO': {'training_freq': 16,
    #           'epoch': 3,
    #           'mini_batch_size': 8,
    #           'gamma': 0.99,
    #           'lam': 0.95,
    #           'epsilon_clip': 0.2,
    #           'value_loss_coef': 0.5,
    #           'entropy_coef': 0.001,
    #           'grad_clip_max_norm': 5.0}}}
        
            
    #     `env_config` YAML Format::
        
    #         num_envs: 3
    #         PPO:
    #           training_freq: 16
    #           epoch: 3
    #           mini_batch_size: 8
    #           gamma: 0.99
    #           lam: 0.95
    #           epsilon_clip: 0.2
    #           value_loss_coef: 0.5
    #           entropy_coef: 0.001
    #           grad_clip_max_norm: 5.0
    #     """
    #     num_envs = env_config["num_envs"]
    #     ppo_config = RecurrentPPOConfig(**env_config["RecurrentPPO"])
    #     return RecurrentPPO(ppo_config, network, policy, num_envs)
        
        
    def update(self, experience: Experience):
        super().update(experience)
        
        self.prev_terminated = torch.from_numpy(experience.terminated)
        
        # add the experience
        self.trajectory.add(
            experience,
            self.current_action_log_prob,
            self.v_pred,
            self.current_hidden_state
        )
        
        # if training frequency is reached, start training
        if self.trajectory.count == self.config.training_freq:
            self.train()
            
    def select_action_train(self, obs: torch.Tensor) -> ActionTensor:
        with torch.no_grad():
            self.current_hidden_state = self.next_hidden_state * (1.0 - self.prev_terminated)
            # when interacting with environment, sequence_length must be 1
            # feed forward
            pdparam, v_pred, hidden_state = self.network.forward(obs.unsqueeze(1), self.current_hidden_state.to(device=self.device))
            
            # action sampling
            dist = self.policy.get_policy_distribution(pdparam)
            action = dist.sample()
            
            # store data
            self.current_action_log_prob = dist.log_prob(action).cpu()
            self.v_pred = v_pred.cpu()
            self.next_hidden_state = hidden_state.cpu()
            
            return action
    
    def select_action_inference(self, obs: torch.Tensor) -> ActionTensor:
        # pdparam, _, _ = self.network.forward(obs)
        # dist = self.policy.get_policy_distribution(pdparam)
        # return dist.sample()
        raise NotImplementedError
            
    def train(self):
        exp_batch = self.trajectory.sample(self.device)
        
        advantage, v_target = self.compute_adavantage_v_target(exp_batch)
        
        obs, discrete_action, continuous_action, old_action_log_prob, advantage, v_target, mask, sequence_start_hidden_state = self.to_batch_sequences(
            exp_batch.obs,
            exp_batch.action,
            exp_batch.terminated,
            exp_batch.action_log_prob,
            advantage,
            v_target,
            exp_batch.hidden_state,
            exp_batch.n_steps
        )
        
        num_sequences = len(obs)
        
        for _ in range(self.config.epoch):
            sample_sequences = torch.randperm(num_sequences)
            for i in range(num_sequences // self.config.num_sequences_per_step):
                sample_sequence = sample_sequences[self.config.num_sequences_per_step * i : self.config.num_sequences_per_step * (i + 1)]
                m = mask[sample_sequence]
                
                # feed forward
                pdparam, v_pred, _ = self.network.forward(obs[sample_sequence], sequence_start_hidden_state[:, sample_sequence])
                
                # compute actor loss
                dist = self.policy.get_policy_distribution(pdparam)
                a = ActionTensor(
                    discrete_action[sample_sequence].flatten(0, 1),
                    continuous_action[sample_sequence].flatten(0, 1)
                )
                new_action_log_prob = dist.log_prob(a).reshape(self.config.num_sequences_per_step, -1, a.num_branches)
                actor_loss = PPO.compute_actor_loss(
                    advantage[sample_sequence][m],
                    old_action_log_prob[sample_sequence][m],
                    new_action_log_prob[m], # maybe cause problem
                    self.config.epsilon_clip
                )
                entropy = dist.entropy().reshape(self.config.num_sequences_per_step, -1, a.num_branches)[m].mean()
                
                # compute critic loss
                v_pred = v_pred.reshape(self.config.num_sequences_per_step, -1, 1)
                critic_loss = PPO.compute_critic_loss(v_pred[m], v_target[sample_sequence][m])
                
                # train step
                loss = actor_loss + self.config.value_loss_coef * critic_loss - self.config.entropy_coef * entropy
                self.network.train_step(loss, self.config.grad_clip_max_norm, self.clock.training_step)
                
                self.clock.tick_training_step()
                
                # log data
                self.actor_average_loss.update(actor_loss.item())
                self.critic_average_loss.update(critic_loss.item())
                
    def to_batch_sequences(self, 
                           obs: torch.Tensor, 
                           action: ActionTensor,
                           terminated: torch.Tensor,
                           action_log_prob: torch.Tensor,
                           advantage: torch.Tensor,
                           v_target: torch.Tensor,
                           hidden_state: torch.Tensor,
                           n_steps: int):
        # 1. stack sequence_length experiences
        # 2. when episode is terminated or remained experiences < sequence_length, zero padding
        # 3. feed forward
        
        # batch_size = 128
        # sequence_length = 8
        # if not teraminted
        # stack experiences
        # if sequence_length is reached or terminated:
        # stop stacking and go to next sequence
        mask = torch.ones((self.num_envs, n_steps))
        
        b2e = lambda x: drl_util.batch2perenv(x, self.num_envs)
        obs = b2e(obs)
        discrete_action = b2e(action.discrete_action) if action.num_discrete_branches > 0 else torch.empty((self.num_envs, n_steps, 0))
        continuous_action = b2e(action.continuous_action) if action.num_continuous_branches > 0 else torch.empty((self.num_envs, n_steps, 0))
        terminated = b2e(terminated)
        action_log_prob = b2e(action_log_prob)
        advantage = b2e(advantage)
        v_target = b2e(v_target)
        # (max_num_layers, batch_size, *out_features) -> (batch_size, max_num_layers, *out_features)
        hidden_state = hidden_state.swapaxes(0, 1)
        hidden_state = b2e(hidden_state)
        
        sequence_start_hidden_state = []
        stacked_obs = []
        stacked_discrete_action = []
        stacked_continuous_action = []
        stacked_action_log_prob = []
        stacked_advantage = []
        stacked_v_target = []
        stacked_mask = []
        
        seq_len = self.config.sequence_length
        
        for env_id in range(self.num_envs):
            seq_start = 0
            t = 0
            terminated_idxes = torch.where(terminated[env_id] > 0.5)[0]
            
            while seq_start < n_steps:
                seq_end = min(seq_start + seq_len, n_steps)
                
                # if terminated in the middle of sequence
                # it will be zero padded
                if t < len(terminated_idxes) and terminated_idxes[t] < seq_end:
                    seq_end = terminated_idxes[t].item() + 1
                    t += 1
                    
                sequence_start_hidden_state.append(hidden_state[env_id, seq_start])
                
                idx = torch.arange(seq_start, seq_end)
                stacked_obs.append(obs[env_id, idx])
                stacked_discrete_action.append(discrete_action[env_id, idx])
                stacked_continuous_action.append(continuous_action[env_id, idx])
                stacked_action_log_prob.append(action_log_prob[env_id, idx])
                stacked_advantage.append(advantage[env_id, idx])
                stacked_v_target.append(v_target[env_id, idx])
                stacked_mask.append(mask[env_id, idx])
                
                seq_start = seq_end

        # (max_num_layers, *out_features) x num_sequences -> (num_sequences, max_num_layers, *out_features)
        sequence_start_hidden_state = torch.stack(sequence_start_hidden_state)
        # (num_sequences, max_num_layers, *out_features) -> (max_num_layers, num_sequences, *out_features)
        sequence_start_hidden_state.swapaxes_(0, 1)
        
        pad = lambda x: pad_sequence(x, batch_first=True)
        stacked_obs = pad(stacked_obs)
        stacked_discrete_action = pad(stacked_discrete_action)
        stacked_continuous_action = pad(stacked_continuous_action)
        stacked_action_log_prob = pad(stacked_action_log_prob)
        stacked_advantage = pad(stacked_advantage)
        stacked_v_target = pad(stacked_v_target)
        stacked_mask = pad(stacked_mask) > 0.5
        
        return stacked_obs, stacked_discrete_action, stacked_continuous_action, stacked_action_log_prob, stacked_advantage, stacked_v_target, stacked_mask, sequence_start_hidden_state

    def compute_adavantage_v_target(self, exp_batch: RecurrentPPOExperienceBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantage, v_target. `batch_size` is `num_evns` x `n_steps`.

        Args:
            exp_batch (PPOExperienceBatch): experience batch

        Returns:
            Tuple[Tensor, Tensor]: advantage, v_target whose each shape is `(batch_size, 1)`
        """
        with torch.no_grad():
            final_next_obs = exp_batch.next_obs[-self.num_envs:]
            final_hidden_state = self.next_hidden_state
            _, final_next_v_pred, _ = self.network.forward(final_next_obs.unsqueeze(1), final_hidden_state.to(device=self.device))
        
        v_pred = torch.cat([exp_batch.v_pred, final_next_v_pred])
        
        # (num_envs * n + 1, 1) -> (num_envs, n, 1) -> (num_envs, n)
        v_pred = drl_util.batch2perenv(v_pred, self.num_envs).squeeze_(-1)
        reward = drl_util.batch2perenv(exp_batch.reward, self.num_envs).squeeze_(-1)
        terminated = drl_util.batch2perenv(exp_batch.terminated, self.num_envs).squeeze_(-1)
        
        # compute advantage using GAE
        advantage = drl_util.compute_gae(
            v_pred,
            reward,
            terminated,
            self.config.gamma,
            self.config.lam
        )
        
        # compute v_target
        v_target = advantage + v_pred[:, :-1]
        
        advantage = drl_util.perenv2batch(advantage.unsqueeze_(-1))
        v_target = drl_util.perenv2batch(v_target.unsqueeze_(-1))
        
        return advantage, v_target

    @property
    def log_keys(self) -> Tuple[str, ...]:
        return super().log_keys + ("Network/Actor Loss", "Network/Critic Loss")
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        ld = super().log_data
        if self.actor_average_loss.count > 0:
            ld["Network/Actor Loss"] = (self.actor_average_loss.average, self.clock.training_step)
            ld["Network/Critic Loss"] = (self.critic_average_loss.average, self.clock.training_step)
            self.actor_average_loss.reset()
            self.critic_average_loss.reset()
        return ld
