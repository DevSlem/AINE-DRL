from typing import Dict, NamedTuple, Optional, Tuple
from aine_drl.agent import Agent
from aine_drl.experience import ActionTensor, Experience
from aine_drl.network import RecurrentActorCriticSharedRNDNetwork
from aine_drl.policy.policy import Policy
from .ppo import PPO
import aine_drl.agent.ppo.ppo_trajectory as tj
import aine_drl.drl_util as drl_util
import aine_drl.util as util
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import math

class RecurrentPPORNDConfig(NamedTuple):
    """
    Recurrent PPO with RND configurations.

    Args:
        `training_freq (int)`: training frequency which is the number of time steps to gather experiences
        `epoch (int)`: number of using total experiences to update parameters at each training frequency
        `sequence_length (int)`: sequence length of recurrent network when training. trajectory is split by `sequence_length` unit. a value of `8` or greater are typically recommended.
        `num_sequences_per_step (int)`: number of sequences per train step, which are selected randomly
        `padding_value (float, optional)`: pad sequences to the value for the same `sequence_length`. Defaults to 0.
        `extrinsic_gamma (float, optional)`: discount factor of extrinsic reward. Defaults to 0.99.
        `intrinsic_gamma (float, optional)`: discount factor of intrinsic reward. Defaults to 0.99.
        `extrinsic_adv_coef (float, optional)`: multiplier to extrinsic advantage. Defaults to 1.0.
        `intrinsic_adv_coef (float, optional)`: multiplier to intrinsic advantage. Defaults to 1.0.
        `lam (float, optional)`: regularization parameter which controls the balanace of Generalized Advantage Estimation (GAE) between bias and variance. Defaults to 0.95.
        `epsilon_clip (float, optional)`: clipping the probability ratio (pi_theta / pi_theta_old) to [1-eps, 1+eps]. Defaults to 0.2.
        `value_loss_coef (float, optional)`: state value loss (critic loss) multiplier. Defaults to 0.5.
        `entropy_coef (float, optional)`: entropy multiplier used to compute loss. It adjusts exploration/exploitation balance. Defaults to 0.001.
        `exp_proportion_for_predictor (float, optional)`: proportion of experience used for training predictor to keep the effective batch size. Defaults to 0.25.
        `pre_normalization_step (int | None, optional)`: number of initial steps for initializing both observation and hidden state normalization. Defaults to no normalization.
        `obs_norm_clip_range (Tuple[float, float])`: observation normalization clipping range (min, max). Defaults to (-5.0, 5.0).
        `hidden_state_norm_clip_range (Tuple[float, float])`: hidden state normalization clipping range (min, max). Defaults to (-5.0, 5.0).
        `grad_clip_max_norm (float | None, optional)`: maximum norm for the gradient clipping. Defaults to no gradient clipping.
    """
    training_freq: int
    epoch: int
    sequence_length: int
    num_sequences_per_step: int
    padding_value: float = 0.0
    extrinsic_gamma: float = 0.99
    intrinsic_gamma: float = 0.99
    extrinsic_adv_coef: float = 1.0
    intrinsic_adv_coef: float = 1.0
    lam: float = 0.95
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    exp_proportion_for_predictor: float = 0.25
    pre_normalization_step: Optional[int] = None
    obs_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
    hidden_state_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
    grad_clip_max_norm: Optional[float] = None
    

class RecurrentPPORND(Agent):
    """
    Recurrent Proximal Policy Optimization (PPO) with Random Network Distillation (RND). \\
    PPO paper: https://arxiv.org/abs/1707.06347 \\
    RND paper: https://arxiv.org/abs/1810.12894

    Args:
        config (RecurrentPPORNDConfig): Recurrent PPO with RND configuration
        network (RecurrentActorCriticSharedRNDNetwork): Recurrent (e.g., LSTM, GRU) Actor Critic Shared RND Network
        policy (Policy): policy
        num_envs (int): number of environments
    """
    def __init__(self, 
                 config: RecurrentPPORNDConfig,
                 network: RecurrentActorCriticSharedRNDNetwork,
                 policy: Policy,
                 num_envs: int) -> None:        
        if not isinstance(network, RecurrentActorCriticSharedRNDNetwork):
            raise TypeError("The network type must be RecurrentActorCriticSharedNetwork.")
        
        super().__init__(network, policy, num_envs)
        
        self.config = config
        self.network = network
        self.trajectory = tj.RecurrentPPORNDTrajectory(self.config.training_freq)
        
        self.current_action_log_prob = None
        self.ext_state_value = None
        self.int_state_value = None
        self.prev_discounted_int_return = 0.0
        self.current_hidden_state = np.zeros(network.hidden_state_shape(self.num_envs), dtype=np.float32)
        self.next_hidden_state = np.zeros(network.hidden_state_shape(self.num_envs), dtype=np.float32)
        self.prev_terminated = np.zeros((self.num_envs, 1), dtype=np.float32)
        self.int_reward_mean_var = util.IncrementalMeanVarianceFromBatch(axis=1) # compute intrinic reward normalization parameters of each env along time steps
        self.next_obs_feature_mean_var = util.IncrementalMeanVarianceFromBatch(axis=0) # compute normalization parameters of each feature of next observation along batches
        self.next_hidden_state_feature_mean_var = util.IncrementalMeanVarianceFromBatch(axis=0) # compute normalization parameters of each feature of next hidden state along batches
        self.current_pre_obs_norm_step = 0
        
        self.actor_average_loss = util.IncrementalAverage()
        self.ext_critic_average_loss = util.IncrementalAverage()
        self.int_critic_average_loss = util.IncrementalAverage()
        self.rnd_average_loss = util.IncrementalAverage()
        self.average_intrinsic_reward = util.IncrementalAverage()
        
        # for inference mode
        self.inference_current_hidden_state = np.zeros(network.hidden_state_shape(1), dtype=np.float32)
        self.inference_next_hidden_state = np.zeros(network.hidden_state_shape(1), dtype=np.float32)
        self.inference_prev_terminated = np.zeros((1, 1), dtype=np.float32)
        
    def update(self, experience: Experience):
        super().update(experience)
        
        self.prev_terminated = experience.terminated
        
        # (D x num_layers, num_envs, H) -> (num_envs, D x num_layers, H)
        next_hidden_state = (self.next_hidden_state * (1.0 - self.prev_terminated)).swapaxes(0, 1)
        
        # pre-normalization
        if (self.config.pre_normalization_step is not None) and (self.current_pre_obs_norm_step < self.config.pre_normalization_step):
            self.current_pre_obs_norm_step += 1
            self.next_obs_feature_mean_var.update(experience.next_obs)
            self.next_hidden_state_feature_mean_var.update(next_hidden_state)
            return
        
        # compute intrinsic reward
        normalized_next_obs = self._normalize_next_obs(torch.from_numpy(experience.next_obs).to(device=self.device))
        normalized_next_hidden_state = self._normalize_next_hidden_state(torch.from_numpy(next_hidden_state).to(device=self.device))
        int_reward = self._compute_intrinsic_reward(normalized_next_obs, normalized_next_hidden_state).cpu().numpy()
        
        # add one experience
        exp_dict = experience._asdict()
        exp_dict["int_reward"] = int_reward
        exp_dict["action_log_prob"] = self.current_action_log_prob
        exp_dict["ext_state_value"] = self.ext_state_value
        exp_dict["int_state_value"] = self.int_state_value
        exp_dict["hidden_state"] = self.current_hidden_state
        self.trajectory.add(tj.RecurrentPPORNDExperience(**exp_dict))
        
        # if training frequency is reached, start training
        if self.trajectory.can_train:
            self._train()
            
    def inference(self, experience: Experience):
        self.inference_prev_terminated = experience.terminated
            
    def select_action_train(self, obs: torch.Tensor) -> ActionTensor:
        with torch.no_grad():
            self.current_hidden_state = self.next_hidden_state * (1.0 - self.prev_terminated)
            # when interacting with environment, sequence_length must be 1
            # feed forward
            pdparam, ext_state_value, int_state_value, next_hidden_state = self.network.actor_critic_net.forward(
                obs.unsqueeze(1), 
                torch.from_numpy(self.current_hidden_state).to(device=self.device)
            )
            
            # action sampling
            dist = self.policy.get_policy_distribution(pdparam)
            action = dist.sample()
            
            # store data
            self.current_action_log_prob = dist.log_prob(action).cpu().numpy()
            self.ext_state_value = ext_state_value.cpu().numpy()
            self.int_state_value = int_state_value.cpu().numpy()
            self.next_hidden_state = next_hidden_state.cpu().numpy()
            
            return action
    
    def select_action_inference(self, obs: torch.Tensor) -> ActionTensor:
        self.inference_current_hidden_state = self.inference_next_hidden_state * (1.0 - self.inference_prev_terminated)
        pdparam, _, _, next_hidden_state = self.network.actor_critic_net.forward(
            obs.unsqueeze(1), 
            torch.from_numpy(self.inference_current_hidden_state).to(device=self.device)
        )
        dist = self.policy.get_policy_distribution(pdparam)
        action = dist.sample()
        self.inference_next_hidden_state = next_hidden_state.cpu().numpy()
        return action
            
    def _train(self):
        exp_batch = self.trajectory.sample(self.device)
        
        # get next hidden state
        final_next_hidden_state = torch.from_numpy(self.next_hidden_state * (1.0 - self.prev_terminated)).to(device=self.device)
        next_hidden_state = torch.concat([exp_batch.hidden_state[:, self.num_envs:], final_next_hidden_state], dim=1)
        
        # compute advantage and target state value
        advantage, ext_target_state_value, int_target_state_value = self._compute_adavantage_v_target(exp_batch)
        
        # convert batch to sequences
        (obs, 
         next_obs, 
         discrete_action, 
         continuous_action, 
         old_action_log_prob, 
         advantage, 
         ext_target_state_value, 
         int_target_state_value, 
         next_hidden_state,
         mask, 
         sequence_start_hidden_state) = self._to_sequences(
            exp_batch.obs,
            exp_batch.next_obs,
            exp_batch.action,
            exp_batch.terminated,
            exp_batch.action_log_prob,
            advantage,
            ext_target_state_value,
            int_target_state_value,
            exp_batch.hidden_state,
            next_hidden_state,
            exp_batch.n_steps
        )
        
        num_sequences = len(obs)
        
        # update next observation and next hidden state normalization parameters
        masked_next_obs = next_obs[mask]
        masked_next_hidden_state = next_hidden_state[mask]
        self.next_obs_feature_mean_var.update(masked_next_obs.cpu().numpy())
        self.next_hidden_state_feature_mean_var.update(masked_next_hidden_state.cpu().numpy())
        
        # normalize next observation and next hidden state
        normalized_next_obs = next_obs.clone()
        normalized_next_hidden_state = next_hidden_state.clone()
        normalized_next_obs[mask] = self._normalize_next_obs(masked_next_obs)
        normalized_next_hidden_state[mask] = self._normalize_next_hidden_state(masked_next_hidden_state)
        
        for _ in range(self.config.epoch):
            sample_sequences = torch.randperm(num_sequences)
            for i in range(num_sequences // self.config.num_sequences_per_step):
                sample_sequence = sample_sequences[self.config.num_sequences_per_step * i : self.config.num_sequences_per_step * (i + 1)]
                m = mask[sample_sequence]
                
                # feed forward
                pdparam, ext_state_value, int_state_value, _ = self.network.actor_critic_net.forward(obs[sample_sequence], sequence_start_hidden_state[:, sample_sequence])
                predicted_feature, target_feature = self.network.rnd_net.forward(
                    normalized_next_obs[sample_sequence][m],
                    normalized_next_hidden_state[sample_sequence][m].flatten(1, 2)
                )
                
                # compute actor loss
                dist = self.policy.get_policy_distribution(pdparam)
                a = ActionTensor(
                    discrete_action[sample_sequence].flatten(0, 1),
                    continuous_action[sample_sequence].flatten(0, 1)
                )
                new_action_log_prob = dist.log_prob(a).reshape(self.config.num_sequences_per_step, -1, 1)
                actor_loss = PPO.compute_actor_loss(
                    advantage[sample_sequence][m],
                    old_action_log_prob[sample_sequence][m],
                    new_action_log_prob[m],
                    self.config.epsilon_clip
                )
                entropy = dist.entropy().reshape(self.config.num_sequences_per_step, -1, 1)[m].mean()
                
                # compute critic loss
                ext_state_value = ext_state_value.reshape(self.config.num_sequences_per_step, -1, 1)
                int_state_value = int_state_value.reshape(self.config.num_sequences_per_step, -1, 1)
                ext_critic_loss = PPO.compute_critic_loss(ext_state_value[m], ext_target_state_value[sample_sequence][m])
                int_critic_loss = PPO.compute_critic_loss(int_state_value[m], int_target_state_value[sample_sequence][m])
                critic_loss = ext_critic_loss + int_critic_loss
                
                # compute RND loss
                rnd_loss = F.mse_loss(predicted_feature, target_feature, reduction="none").mean(dim=-1)
                # proportion of exp used for predictor update
                rnd_loss_mask = torch.rand(len(rnd_loss), device=rnd_loss.device)
                rnd_loss_mask = (rnd_loss_mask < self.config.exp_proportion_for_predictor).to(dtype=rnd_loss.dtype)
                rnd_loss = (rnd_loss * rnd_loss_mask).sum() / torch.max(rnd_loss_mask.sum(), torch.tensor(1.0, device=rnd_loss.device))
                
                # train step
                loss = actor_loss + self.config.value_loss_coef * critic_loss - self.config.entropy_coef * entropy + rnd_loss
                self.network.train_step(loss, self.config.grad_clip_max_norm, self.clock.training_step)
                self.clock.tick_training_step()
                
                # log data
                self.actor_average_loss.update(actor_loss.item())
                self.ext_critic_average_loss.update(ext_critic_loss.item())
                self.int_critic_average_loss.update(int_critic_loss.item())
                self.rnd_average_loss.update(rnd_loss.item())
                
    def _to_sequences(self, 
                           obs: torch.Tensor,
                           next_obs: torch.Tensor,
                           action: ActionTensor,
                           terminated: torch.Tensor,
                           action_log_prob: torch.Tensor,
                           advantage: torch.Tensor,
                           ext_target_state_value: torch.Tensor,
                           int_target_state_value: torch.Tensor,
                           hidden_state: torch.Tensor,
                           next_hidden_state: torch.Tensor,
                           n_steps: int) -> Tuple[torch.Tensor, ...]:
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
        next_obs = b2e(next_obs)
        discrete_action = b2e(action.discrete_action) if action.num_discrete_branches > 0 else torch.empty((self.num_envs, n_steps, 0))
        continuous_action = b2e(action.continuous_action) if action.num_continuous_branches > 0 else torch.empty((self.num_envs, n_steps, 0))
        terminated = b2e(terminated)
        action_log_prob = b2e(action_log_prob)
        advantage = b2e(advantage)
        ext_target_state_value = b2e(ext_target_state_value)
        int_target_state_value = b2e(int_target_state_value)
        # (D x num_layers, batch_size, H) -> (batch_size, D x num_layers, H)
        hidden_state = hidden_state.swapaxes(0, 1)
        next_hidden_state = next_hidden_state.swapaxes(0, 1)
        hidden_state = b2e(hidden_state)
        next_hidden_state = b2e(next_hidden_state)
        
        sequence_start_hidden_state = []
        stacked_obs = []
        stacked_next_obs = []
        stacked_discrete_action = []
        stacked_continuous_action = []
        stacked_action_log_prob = []
        stacked_advantage = []
        stacked_ext_target_state_value = []
        stacked_int_target_state_value = []
        stacked_next_hidden_state = []
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
                sequence_interrupted = t < len(terminated_idxes) and terminated_idxes[t] < seq_end
                if sequence_interrupted:
                    seq_end = terminated_idxes[t].item() + 1
                    t += 1
                    
                sequence_start_hidden_state.append(hidden_state[env_id, seq_start])
                
                idx = torch.arange(seq_start, seq_end)
                stacked_obs.append(obs[env_id, idx])
                stacked_next_obs.append(next_obs[env_id, idx])
                stacked_discrete_action.append(discrete_action[env_id, idx])
                stacked_continuous_action.append(continuous_action[env_id, idx])
                stacked_action_log_prob.append(action_log_prob[env_id, idx])
                stacked_advantage.append(advantage[env_id, idx])
                stacked_ext_target_state_value.append(ext_target_state_value[env_id, idx])
                stacked_int_target_state_value.append(int_target_state_value[env_id, idx])
                next_hidden_state_sequence = next_hidden_state[env_id, idx]
                # when terminated, next_hidden_state is zero
                if sequence_interrupted:
                    next_hidden_state_sequence[-1] = torch.zeros_like(next_hidden_state_sequence[-1])
                stacked_next_hidden_state.append(next_hidden_state_sequence)
                stacked_mask.append(mask[env_id, idx])
                
                seq_start = seq_end

        # (D x num_layers, *out_features) x num_sequences -> (num_sequences, D x num_layers, *out_features)
        sequence_start_hidden_state = torch.stack(sequence_start_hidden_state)
        # (num_sequences, D x num_layers, *out_features) -> (D x num_layers, num_sequences, *out_features)
        sequence_start_hidden_state.swapaxes_(0, 1)
        
        pad = lambda x: pad_sequence(x, batch_first=True, padding_value=self.config.padding_value)
        
        stacked_obs = pad(stacked_obs)
        stacked_next_obs = pad(stacked_next_obs)
        stacked_discrete_action = pad(stacked_discrete_action)
        stacked_continuous_action = pad(stacked_continuous_action)
        stacked_action_log_prob = pad(stacked_action_log_prob)
        stacked_advantage = pad(stacked_advantage)
        stacked_ext_target_state_value = pad(stacked_ext_target_state_value)
        stacked_int_target_state_value = pad(stacked_int_target_state_value)
        stacked_next_hidden_state = pad(stacked_next_hidden_state)
        stacked_mask = pad(stacked_mask)
        eps = torch.finfo(torch.float32).eps * 2.0
        stacked_mask = (stacked_mask < self.config.padding_value - eps) | (stacked_mask > self.config.padding_value + eps)
        
        return (stacked_obs, 
                stacked_next_obs, 
                stacked_discrete_action, 
                stacked_continuous_action, 
                stacked_action_log_prob, 
                stacked_advantage, 
                stacked_ext_target_state_value, 
                stacked_int_target_state_value, 
                stacked_next_hidden_state,
                stacked_mask, 
                sequence_start_hidden_state)

    def _compute_adavantage_v_target(self, exp_batch: tj.RecurrentPPORNDExperienceBatchTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute advantage, target state values. `batch_size` is `num_evns` x `n_steps`. 
        The shape of each returned tensor is `(batch_size, 1)`.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: advantage, extrinisc target state value, intrinsic target state value
        """
        with torch.no_grad():
            final_next_obs = exp_batch.next_obs[-self.num_envs:].unsqueeze(dim=1)
            final_hidden_state = torch.from_numpy(self.next_hidden_state).to(device=self.device)
            _, final_ext_next_state_value, final_int_next_state_value, _ = self.network.actor_critic_net.forward(
                final_next_obs,
                final_hidden_state
            )
        
        # concate n-step state values and one next state value of final transition
        # shape is (num_envs * n+1, 1)
        ext_state_value = torch.cat([exp_batch.ext_state_value, final_ext_next_state_value])
        int_state_value = torch.cat([exp_batch.int_state_value, final_int_next_state_value])
                
        # (num_envs * t, 1) -> (num_envs, t, 1) -> (num_envs, t)
        b2e = lambda x: drl_util.batch2perenv(x, self.num_envs).squeeze_(-1)
        ext_state_value = b2e(ext_state_value)
        int_state_value = b2e(int_state_value)
        reward = b2e(exp_batch.reward)
        terminated = b2e(exp_batch.terminated)
        int_reward = b2e(exp_batch.int_reward)
        
        # # compute discounted intrinsic return
        discounted_int_return = torch.empty_like(int_reward)
        for t in range(exp_batch.n_steps):
            self.prev_discounted_int_return = int_reward[:, t] + self.config.intrinsic_gamma * self.prev_discounted_int_return
            discounted_int_return[:, t] = self.prev_discounted_int_return
        
        # update intrinic reward normalization parameters
        self.int_reward_mean_var.update(discounted_int_return.cpu().numpy())
        
        # normalize intinrisc reward
        int_reward /= torch.from_numpy(np.sqrt(self.int_reward_mean_var.variance)[..., np.newaxis]).to(device=self.device)
        # self.int_reward_mean_var.update(drl_util.perenv2batch(discounted_int_reward).detach().cpu().numpy())
        # int_reward /= math.sqrt(self.int_reward_mean_var.variance)
        self.average_intrinsic_reward.update(int_reward.mean().item())
        
        # compute advantage using GAE
        ext_advantage = drl_util.compute_gae(
            ext_state_value,
            reward,
            terminated,
            self.config.extrinsic_gamma,
            self.config.lam
        )
        int_advantage = drl_util.compute_gae(
            int_state_value,
            int_reward,
            torch.zeros_like(terminated), # non-episodic
            self.config.intrinsic_gamma,
            self.config.lam
        )
        advantage = self.config.extrinsic_adv_coef * ext_advantage + self.config.intrinsic_adv_coef * int_advantage
        
        # compute target state values
        ext_target_state_value = ext_advantage + ext_state_value[:, :-1]
        int_target_state_value = int_advantage + int_state_value[:, :-1]
        
        # (num_envs, t) -> (num_envs, t, 1) -> (num_envs * t, 1)
        advantage = drl_util.perenv2batch(advantage.unsqueeze_(-1))
        ext_target_state_value = drl_util.perenv2batch(ext_target_state_value.unsqueeze_(-1))
        int_target_state_value = drl_util.perenv2batch(int_target_state_value.unsqueeze_(-1))
        
        return advantage, ext_target_state_value, int_target_state_value
    
    def _compute_intrinsic_reward(self, next_obs: torch.Tensor, next_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward.
        
        Args:
            next_obs (Tensor): `(batch_size, *obs_shape)`
            next_hidden_state (Tensor): `(batch_size, D x num_layers, H)`
            
        Returns:
            intrinsic_reward (Tensor): `(batch_size, 1)`
        """
        with torch.no_grad():
            predicted_feature, target_feature = self.network.rnd_net.forward(next_obs, next_hidden_state.flatten(1, 2))
            intrinsic_reward = 0.5 * ((target_feature - predicted_feature)**2).sum(dim=1, keepdim=True)
            return intrinsic_reward
        
    def _normalize_next_obs(self, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Normalize next observation. If `pre_normalization_step` setting in the configuration is `None`, this method doesn't normalize it.
        """
        if self.config.pre_normalization_step is None:
            return next_obs
        obs_feature_mean = torch.from_numpy(self.next_obs_feature_mean_var.mean).to(dtype=torch.float32, device=next_obs.device)
        obs_feature_std = torch.from_numpy(np.sqrt(self.next_obs_feature_mean_var.variance)).to(dtype=torch.float32, device=next_obs.device)
        normalized_next_obs = (next_obs - obs_feature_mean) / obs_feature_std
        return normalized_next_obs.clip(self.config.obs_norm_clip_range[0], self.config.obs_norm_clip_range[1])

    def _normalize_next_hidden_state(self, next_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Normalize next hidden state. If `pre_normalization_step` setting in the configuration is `None`, this method doesn't normalize it.
        """
        if self.config.pre_normalization_step is None:
            return next_hidden_state
        hidden_state_feature_mean = torch.from_numpy(self.next_hidden_state_feature_mean_var.mean).to(dtype=torch.float32, device=next_hidden_state.device)
        hidden_state_feature_std = torch.from_numpy(np.sqrt(self.next_hidden_state_feature_mean_var.variance)).to(dtype=torch.float32, device=next_hidden_state.device)
        normalized_next_hidden_state = (next_hidden_state - hidden_state_feature_mean) / hidden_state_feature_std
        return normalized_next_hidden_state.clip(self.config.hidden_state_norm_clip_range[0], self.config.hidden_state_norm_clip_range[1])
    
    @property
    def log_keys(self) -> Tuple[str, ...]:
        return super().log_keys + ("Network/Actor Loss", "Network/Extrinsic Critic Loss", "Network/Intrinsic Critic Loss", "Network/RND Loss", "RND/Intrinsic Reward")
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        ld = super().log_data
        if self.actor_average_loss.count > 0:
            ld["Network/Actor Loss"] = (self.actor_average_loss.average, self.clock.training_step)
            ld["Network/Extrinsic Critic Loss"] = (self.ext_critic_average_loss.average, self.clock.training_step)
            ld["Network/Intrinsic Critic Loss"] = (self.int_critic_average_loss.average, self.clock.training_step)
            ld["Network/RND Loss"] = (self.rnd_average_loss.average, self.clock.training_step)
            self.actor_average_loss.reset()
            self.ext_critic_average_loss.reset()
            self.int_critic_average_loss.reset()
            self.rnd_average_loss.reset()
        if self.average_intrinsic_reward.count > 0:
            ld["RND/Intrinsic Reward"] = (self.average_intrinsic_reward.average, self.clock.global_time_step)
            self.average_intrinsic_reward.reset()
        return ld
