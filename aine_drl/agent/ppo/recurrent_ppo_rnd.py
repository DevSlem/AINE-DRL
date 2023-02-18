from aine_drl.agent import Agent
from aine_drl.experience import ActionTensor, Experience
from aine_drl.network import NetworkTypeError
from aine_drl.policy.policy import Policy
from .config import RecurrentPPORNDConfig
from .net import RecurrentPPORNDNetwork
from .ppo import PPO
import aine_drl.agent.ppo.ppo_trajectory as tj
import aine_drl.drl_util as drl_util
import aine_drl.util as util
import torch
import torch.nn.functional as F
import numpy as np

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
                 network: RecurrentPPORNDNetwork,
                 policy: Policy,
                 num_envs: int) -> None:        
        if not isinstance(network, RecurrentPPORNDNetwork):
            raise NetworkTypeError(RecurrentPPORNDNetwork)
        
        super().__init__(network, policy, num_envs)
        
        self.config = config
        self.network = network
        self.trajectory = tj.RecurrentPPORNDTrajectory(self.config.training_freq)
        
        self.current_action_log_prob = None
        self.ext_state_value = None
        self.int_state_value = None
        self.prev_discounted_int_return = 0.0
        hidden_state_shape = (network.hidden_state_shape[0], self.num_envs, network.hidden_state_shape[1])
        self.current_hidden_state = np.zeros(hidden_state_shape, dtype=np.float32)
        self.next_hidden_state = np.zeros(hidden_state_shape, dtype=np.float32)
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
        inference_hidden_state_shape = (network.hidden_state_shape[0], 1, network.hidden_state_shape[1])
        self.inference_current_hidden_state = np.zeros(inference_hidden_state_shape, dtype=np.float32)
        self.inference_next_hidden_state = np.zeros(inference_hidden_state_shape, dtype=np.float32)
        self.inference_prev_terminated = np.zeros((1, 1), dtype=np.float32)
        
    @property
    def name(self) -> str:
        return "Recurrent PPO RND"
        
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
            
            # feed forward
            # when interacting with environment, sequence_length must be 1
            # *batch_shape = (num_seq, seq_len) = (num_envs, 1)
            pdparam_seq, ext_state_value_seq, int_state_value_seq, next_hidden_state = self.network.forward_actor_critic(
                obs.unsqueeze(dim=1), 
                torch.from_numpy(self.current_hidden_state).to(device=self.device)
            )
            
            # action sampling
            dist = self.policy.get_policy_distribution(pdparam_seq)
            action = dist.sample()
            
            # store data
            self.current_action_log_prob = dist.joint_log_prob(action).squeeze_(dim=1).cpu().numpy()
            self.ext_state_value = ext_state_value_seq.squeeze_(dim=1).cpu().numpy()
            self.int_state_value = int_state_value_seq.squeeze_(dim=1).cpu().numpy()
            self.next_hidden_state = next_hidden_state.cpu().numpy()
            
            return action.transform(lambda a: a.squeeze_(dim=1))
    
    def select_action_inference(self, obs: torch.Tensor) -> ActionTensor:
        self.inference_current_hidden_state = self.inference_next_hidden_state * (1.0 - self.inference_prev_terminated)
        pdparam_seq, _, _, next_hidden_state = self.network.forward_actor_critic(
            obs.unsqueeze(dim=1), 
            torch.from_numpy(self.inference_current_hidden_state).to(device=self.device)
        )
        dist = self.policy.get_policy_distribution(pdparam_seq)
        action = dist.sample()
        self.inference_next_hidden_state = next_hidden_state.cpu().numpy()
        return action.transform(lambda a: a.squeeze_(dim=1))
            
    def _train(self):
        exp_batch = self.trajectory.sample(self.device)
        
        # get next hidden state
        final_next_hidden_state = torch.from_numpy(self.next_hidden_state * (1.0 - self.prev_terminated)).to(device=self.device)
        next_hidden_state = torch.concat([exp_batch.hidden_state[:, self.num_envs:], final_next_hidden_state], dim=1)
        
        # compute advantage and target state value
        advantage, ext_target_state_value, int_target_state_value = self._compute_adavantage_v_target(exp_batch)
        
        # convert batch to truncated sequence
        seq_generator = drl_util.TruncatedSequenceGenerator(self.config.sequence_length, self.num_envs, exp_batch.n_steps, self.config.padding_value)
        
        def add_to_seq_gen(batch, start_idx = 0, seq_len = 0):
            seq_generator.add(drl_util.batch2perenv(batch, self.num_envs), start_idx=start_idx, seq_len=seq_len)
            
        add_to_seq_gen(exp_batch.hidden_state.swapaxes(0, 1), seq_len=1)
        add_to_seq_gen(next_hidden_state.swapaxes(0, 1))
        add_to_seq_gen(exp_batch.obs)
        add_to_seq_gen(exp_batch.next_obs)
        if exp_batch.action.num_discrete_branches > 0:
            add_to_seq_gen(exp_batch.action.discrete_action)
        else:
            seq_generator.add(torch.empty((self.num_envs, exp_batch.n_steps, 0)))
        if exp_batch.action.num_continuous_branches > 0:
            add_to_seq_gen(exp_batch.action.continuous_action)
        else:
            seq_generator.add(torch.empty((self.num_envs, exp_batch.n_steps, 0)))
        add_to_seq_gen(exp_batch.action_log_prob)
        add_to_seq_gen(advantage)
        add_to_seq_gen(ext_target_state_value)
        add_to_seq_gen(int_target_state_value)
        
        sequences = seq_generator.generate(drl_util.batch2perenv(exp_batch.terminated, self.num_envs).unsqueeze_(-1))
        (mask, seq_init_hidden_state, next_hidden_state_seq, obs_seq, next_obs_seq, discrete_action_seq, continuous_action_seq, 
         old_action_log_prob_seq, advantage_seq, ext_target_state_value_seq, int_target_state_value_seq) = sequences
        
        num_seq = len(mask)
        # (num_seq, 1, D x num_layers, H) -> (D x num_layers, num_seq, H)
        seq_init_hidden_state = seq_init_hidden_state.squeeze_(dim=1).swapaxes_(0, 1)
        
        # update next observation and next hidden state normalization parameters
        # when masked by mask, (num_seq, seq_len, *shape) -> (masked_batch_size, *shape)
        masked_next_obs = next_obs_seq[mask]
        masked_next_hidden_state = next_hidden_state_seq[mask]
        self.next_obs_feature_mean_var.update(masked_next_obs.cpu().numpy())
        self.next_hidden_state_feature_mean_var.update(masked_next_hidden_state.cpu().numpy())
        
        # normalize next observation and next hidden state
        normalized_next_obs_seq = next_obs_seq
        normalized_next_hidden_state_seq = next_hidden_state_seq
        normalized_next_obs_seq[mask] = self._normalize_next_obs(masked_next_obs)
        normalized_next_hidden_state_seq[mask] = self._normalize_next_hidden_state(masked_next_hidden_state)
        
        for _ in range(self.config.epoch):
            sample_sequences = torch.randperm(num_seq)
            for i in range(num_seq // self.config.num_sequences_per_step):
                # when sliced by sample_seq, (num_seq,) -> (num_seq_per_step,)
                sample_seq = sample_sequences[self.config.num_sequences_per_step * i : self.config.num_sequences_per_step * (i + 1)]
                # when masked by m, (num_seq_per_step, seq_len,) -> (masked_batch_size,)
                m = mask[sample_seq]
                
                # feed forward
                # in this case num_seq = num_seq_per_step
                pdparam_seq, ext_state_value_seq, int_state_value_seq, _ = self.network.forward_actor_critic(
                    obs_seq[sample_seq], 
                    seq_init_hidden_state[:, sample_seq]
                )
                predicted_feature, target_feature = self.network.forward_rnd(
                    normalized_next_obs_seq[sample_seq][m],
                    normalized_next_hidden_state_seq[sample_seq][m].flatten(1, 2)
                )
                
                # compute actor loss
                dist = self.policy.get_policy_distribution(pdparam_seq)
                sample_action_seq = ActionTensor(
                    discrete_action_seq[sample_seq],
                    continuous_action_seq[sample_seq]
                )
                new_action_log_prob_seq = dist.joint_log_prob(sample_action_seq)
                actor_loss = PPO.compute_actor_loss(
                    advantage_seq[sample_seq][m],
                    old_action_log_prob_seq[sample_seq][m],
                    new_action_log_prob_seq[m],
                    self.config.epsilon_clip
                )
                entropy = dist.joint_entropy()[m].mean()
                
                # compute critic loss
                ext_critic_loss = PPO.compute_critic_loss(ext_state_value_seq[m], ext_target_state_value_seq[sample_seq][m])
                int_critic_loss = PPO.compute_critic_loss(int_state_value_seq[m], int_target_state_value_seq[sample_seq][m])
                critic_loss = ext_critic_loss + int_critic_loss
                
                # compute RND loss
                rnd_loss = F.mse_loss(predicted_feature, target_feature, reduction="none").mean(dim=-1)
                # proportion of exp used for predictor update
                rnd_loss_mask = torch.rand(len(rnd_loss), device=rnd_loss.device)
                rnd_loss_mask = (rnd_loss_mask < self.config.exp_proportion_for_predictor).to(dtype=rnd_loss.dtype)
                rnd_loss = (rnd_loss * rnd_loss_mask).sum() / torch.max(rnd_loss_mask.sum(), torch.tensor(1.0, device=rnd_loss.device))
                
                # train step
                loss = actor_loss + self.config.value_loss_coef * critic_loss - self.config.entropy_coef * entropy + rnd_loss
                self.network.train_step(loss, self.clock.training_step)
                self.clock.tick_training_step()
                
                # log data
                self.actor_average_loss.update(actor_loss.item())
                self.ext_critic_average_loss.update(ext_critic_loss.item())
                self.int_critic_average_loss.update(int_critic_loss.item())
                self.rnd_average_loss.update(rnd_loss.item())

    def _compute_adavantage_v_target(self, exp_batch: tj.RecurrentPPORNDExperienceBatchTensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute advantage, extrinisc target state value, intrinsic target state value.
        
        Returns:
            advantage (Tensor): `(num_envs x n_steps, 1)`
            extrinisc target state value (Tensor): `(num_envs x n_steps, 1)`
            intrinsic target state value (Tensor): `(num_envs x n_steps, 1)`
        """
        # (num_envs, *obs_shape)
        final_next_obs = exp_batch.next_obs[-self.num_envs:]
        final_next_hidden_state = torch.from_numpy(self.next_hidden_state).to(device=self.device)
        
        # feed forward without gradient calculation
        with torch.no_grad():
            # (num_envs, 1, *obs_shape) because sequence length is 1
            _, final_ext_next_state_value_seq, final_int_next_state_value_seq, _ = self.network.forward_actor_critic(
                final_next_obs.unsqueeze(dim=1), 
                final_next_hidden_state
            )
        
        # (num_envs, 1, 1) -> (num_envs, 1)
        final_ext_next_state_value = final_ext_next_state_value_seq.squeeze_(dim=1)
        final_int_next_state_value = final_int_next_state_value_seq.squeeze_(dim=1)
        # (num_envs x (n_steps + 1), 1)
        total_ext_state_value = torch.cat([exp_batch.ext_state_value, final_ext_next_state_value])
        total_int_state_value = torch.cat([exp_batch.int_state_value, final_int_next_state_value])
                
        # (num_envs x T, 1) -> (num_envs, T, 1) -> (num_envs, T)
        b2e = lambda x: drl_util.batch2perenv(x, self.num_envs).squeeze_(-1) 
        total_ext_state_value = b2e(total_ext_state_value) # T = n_steps + 1
        total_int_state_value = b2e(total_int_state_value) # T = n_steps + 1
        reward = b2e(exp_batch.reward) # T = n_steps
        terminated = b2e(exp_batch.terminated) # T = n_steps
        int_reward = b2e(exp_batch.int_reward) # T = n_steps
        
        # compute discounted intrinsic return
        discounted_int_return = torch.empty_like(int_reward)
        for t in range(exp_batch.n_steps):
            self.prev_discounted_int_return = int_reward[:, t] + self.config.intrinsic_gamma * self.prev_discounted_int_return
            discounted_int_return[:, t] = self.prev_discounted_int_return
        
        # update intrinic reward normalization parameters
        self.int_reward_mean_var.update(discounted_int_return.cpu().numpy())
        
        # normalize intinrisc reward
        int_reward /= torch.from_numpy(np.sqrt(self.int_reward_mean_var.variance)[..., np.newaxis]).to(device=self.device)
        self.average_intrinsic_reward.update(int_reward.mean().item())
        
        # compute advantage (num_envs, n_steps) using GAE
        ext_advantage = drl_util.compute_gae(
            total_ext_state_value,
            reward,
            terminated,
            self.config.extrinsic_gamma,
            self.config.lam
        )
        int_advantage = drl_util.compute_gae(
            total_int_state_value,
            int_reward,
            torch.zeros_like(terminated), # non-episodic
            self.config.intrinsic_gamma,
            self.config.lam
        )
        advantage = self.config.extrinsic_adv_coef * ext_advantage + self.config.intrinsic_adv_coef * int_advantage
        
        # compute target state values (num_envs, n_steps)
        ext_target_state_value = ext_advantage + total_ext_state_value[:, :-1]
        int_target_state_value = int_advantage + total_int_state_value[:, :-1]
        
        # (num_envs, n_steps) -> (num_envs x n_steps, 1)
        e2b = lambda x: drl_util.perenv2batch(x.unsqueeze_(-1))
        advantage = e2b(advantage)
        ext_target_state_value = e2b(ext_target_state_value)
        int_target_state_value = e2b(int_target_state_value)
        
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
            predicted_feature, target_feature = self.network.forward_rnd(next_obs, next_hidden_state.flatten(1, 2))
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
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/Actor Loss", "Network/Extrinsic Critic Loss", "Network/Intrinsic Critic Loss", "Network/RND Loss", "RND/Intrinsic Reward")
    
    @property
    def log_data(self) -> dict[str, tuple]:
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
