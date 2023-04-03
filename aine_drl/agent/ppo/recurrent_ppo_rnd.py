import torch

import aine_drl.rl_loss as L
import aine_drl.util as util
from aine_drl.agent.agent import Agent, BehaviorType
from aine_drl.agent.ppo.config import RecurrentPPORNDConfig
from aine_drl.agent.ppo.net import RecurrentPPORNDNetwork
from aine_drl.agent.ppo.trajectory import (RecurrentPPORNDExperience,
                                           RecurrentPPORNDTrajectory)
from aine_drl.exp import Action, Experience, Observation
from aine_drl.net import NetworkTypeError, Trainer
from aine_drl.util.func import batch2perenv, perenv2batch


class RecurrentPPORND(Agent):
    """
    Recurrent Proximal Policy Optimization (PPO) with Random Network Distillation (RND).
    
    PPO paper: https://arxiv.org/abs/1707.06347 \\
    RND paper: https://arxiv.org/abs/1810.12894
    """
    def __init__(
        self, 
        config: RecurrentPPORNDConfig,
        network: RecurrentPPORNDNetwork,
        trainer: Trainer,
        num_envs: int,
        behavior_type: BehaviorType = BehaviorType.TRAIN,
    ) -> None:        
        if not isinstance(network, RecurrentPPORNDNetwork):
            raise NetworkTypeError(RecurrentPPORNDNetwork)
        
        super().__init__(num_envs, network, config.device, behavior_type)
        
        self._config = config
        self._network = network
        self._trainer = trainer
        self._trajectory = RecurrentPPORNDTrajectory(self._config.n_steps)
        
        self._action_log_prob: torch.Tensor = None # type: ignore
        self._ext_state_value: torch.Tensor = None # type: ignore
        self._int_state_value: torch.Tensor = None # type: ignore
        self._prev_discounted_int_return = 0.0
        hidden_state_shape = (network.hidden_state_shape()[0], self._num_envs, network.hidden_state_shape()[1])
        self._hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._next_hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._prev_terminated = torch.zeros((self._num_envs, 1), device=self.device)
        # compute intrinic reward normalization parameters of each env along time steps
        self._int_reward_mean_var = util.IncrementalMeanVarianceFromBatch(dim=1, device=self.device) 
        # compute normalization parameters of each feature of next observation along batches
        self._next_obs_feature_mean_var: tuple[util.IncrementalMeanVarianceFromBatch] = tuple()
        # compute normalization parameters of each feature of next hidden state along batches
        self._next_hidden_state_feature_mean_var = util.IncrementalMeanVarianceFromBatch(dim=0, device=self.device) 
        self._current_init_norm_steps = 0
        
        self._actor_loss_mean = util.IncrementalMean()
        self._ext_critic_loss_mean = util.IncrementalMean()
        self._int_critic_loss_mean = util.IncrementalMean()
        self._rnd_loss_mean = util.IncrementalMean()
        self._int_reward_mean = util.IncrementalMean()
        
        # for inference mode
        infer_hidden_state_shape = (network.hidden_state_shape()[0], 1, network.hidden_state_shape()[1])
        self._infer_current_hidden_state = torch.zeros(infer_hidden_state_shape, device=self.device)
        self._infer_next_hidden_state = torch.zeros(infer_hidden_state_shape, device=self.device)
        self._infer_prev_terminated = torch.zeros((1, 1), device=self.device)
        
    @property
    def name(self) -> str:
        return "Recurrent PPO RND"
    
    @property
    def config_dict(self) -> dict:
        return self._config.__dict__
    
    def _update_train(self, exp: Experience):
        self._prev_terminated = exp.terminated
        
        # (D x num_layers, num_envs, H) -> (num_envs, D x num_layers, H)
        next_hidden_state = (self._next_hidden_state * (1.0 - self._prev_terminated)).swapdims(0, 1)
        
        # initialize normalization parameters
        if (self._config.init_norm_steps is not None) and (self._current_init_norm_steps < self._config.init_norm_steps):
            self._current_init_norm_steps += 1
            if len(self._next_obs_feature_mean_var) == 0:
                self._next_obs_feature_mean_var = tuple(
                    util.IncrementalMeanVarianceFromBatch(dim=0, device=self.device) for _ in range(exp.next_obs.num_items)
                )
            for mean_var, obs in zip(self._next_obs_feature_mean_var, exp.next_obs.items):
                mean_var.update(obs)
            self._next_hidden_state_feature_mean_var.update(next_hidden_state)
            return
        
        # compute intrinsic reward
        normalized_next_obs = self._normalize_next_obs(exp.next_obs)
        normalized_next_hidden_state = self._normalize_next_hidden_state(next_hidden_state)
        int_reward = self._compute_intrinsic_reward(normalized_next_obs, normalized_next_hidden_state)
        
        # add one experience
        self._trajectory.add(RecurrentPPORNDExperience(
            obs=exp.obs,
            action=exp.action,
            next_obs=exp.next_obs,
            ext_reward=exp.reward,
            int_reward=int_reward,
            terminated=exp.terminated,
            action_log_prob=self._action_log_prob,
            ext_state_value=self._ext_state_value,
            int_state_value=self._int_state_value,
            hidden_state=self._hidden_state,
            next_hidden_state=self._next_hidden_state
        ))
        
        if self._trajectory.reached_n_steps:
            self._train()
    
    def _update_inference(self, exp: Experience):
        self._infer_prev_terminated = exp.terminated
    
    @torch.no_grad()
    def _select_action_train(self, obs: Observation) -> Action:
        self._hidden_state = self._next_hidden_state * (1.0 - self._prev_terminated)
            
        # feed forward
        # when interacting with environment, sequence_length must be 1
        # *batch_shape = (seq_batch_size, seq_len) = (num_envs, 1)
        policy_dist_seq, ext_state_value_seq, int_state_value_seq, next_hidden_state = self._network.forward_actor_critic(
            obs.transform(lambda o: o.unsqueeze(dim=1)),
            self._hidden_state
        )
        
        # action sampling
        action_seq = policy_dist_seq.sample()
        
        # (num_envs, 1, *shape) -> (num_envs, *shape)
        action = action_seq.transform(lambda a: a.squeeze(dim=1))
        self._action_log_prob = policy_dist_seq.joint_log_prob(action_seq).squeeze_(dim=1)
        self._ext_state_value = ext_state_value_seq.squeeze_(dim=1)
        self._int_state_value = int_state_value_seq.squeeze_(dim=1)
        
        self._next_hidden_state = next_hidden_state
        
        return action
    
    @torch.no_grad()
    def _select_action_inference(self, obs: Observation) -> Action:
        self._infer_hidden_state = self._infer_next_hidden_state * (1.0 - self._infer_prev_terminated)
        policy_dist_seq, _, _, next_hidden_state = self._network.forward_actor_critic(
            obs.transform(lambda o: o.unsqueeze(dim=1)),
            self._infer_hidden_state
        )
        action_seq = policy_dist_seq.sample()
        self._infer_next_hidden_state = next_hidden_state
        return action_seq.transform(lambda a: a.squeeze(dim=1))
            
    def _train(self):
        exp_batch = self._trajectory.sample()
        
        # compute advantage and target state value
        advantage, ext_target_state_value, int_target_state_value = self._compute_adv_target(exp_batch)
        
        # convert batch to truncated sequence
        seq_generator = util.TruncatedSeqGen(
            self._config.seq_len, 
            self._num_envs,
            self._config.n_steps, 
            self._config.padding_value
        )
        
        def add_to_seq_gen(batch, start_idx = 0, seq_len = 0):
            seq_generator.add(batch2perenv(batch, self._num_envs), start_idx=start_idx, seq_len=seq_len)
            
        add_to_seq_gen(exp_batch.hidden_state.swapaxes(0, 1), seq_len=1)
        add_to_seq_gen(exp_batch.next_hidden_state.swapaxes(0, 1))
        for obs in exp_batch.next_obs.items:
            add_to_seq_gen(obs)
        for next_obs in exp_batch.next_obs.items:
            add_to_seq_gen(next_obs)
        if exp_batch.action.num_discrete_branches > 0:
            add_to_seq_gen(exp_batch.action.discrete_action)
        else:
            seq_generator.add(torch.empty((self._num_envs, self._config.n_steps, 0)))
        if exp_batch.action.num_continuous_branches > 0:
            add_to_seq_gen(exp_batch.action.continuous_action)
        else:
            seq_generator.add(torch.empty((self._num_envs, self._config.n_steps, 0)))
        add_to_seq_gen(exp_batch.action_log_prob)
        add_to_seq_gen(advantage)
        add_to_seq_gen(ext_target_state_value)
        add_to_seq_gen(int_target_state_value)
        
        sequences = seq_generator.generate(batch2perenv(exp_batch.terminated, self._num_envs).unsqueeze_(-1))
        
        mask = sequences[0]
        seq_init_hidden_state, next_hidden_state_seq = sequences[1:3]
        
        next_obs_seq_start_idx = 3 + exp_batch.obs.num_items
        obs_seq = Observation(sequences[3:next_obs_seq_start_idx])
        
        action_seq_start_idx = next_obs_seq_start_idx + exp_batch.next_obs.num_items
        next_obs_seq = Observation(sequences[next_obs_seq_start_idx:action_seq_start_idx])
        
        action_seq = Action(*sequences[action_seq_start_idx:action_seq_start_idx + 2])
        old_action_log_prob_seq, advantage_seq, ext_target_state_value_seq, int_target_state_value_seq = sequences[action_seq_start_idx + 2:]
        
        seq_batch_size = len(mask)
        # (seq_batch_size, 1, D x num_layers, H) -> (D x num_layers, seq_batch_size, H)
        seq_init_hidden_state = seq_init_hidden_state.squeeze_(dim=1).swapdims_(0, 1)
        
        # update next observation and next hidden state normalization parameters
        # when masked by mask, (seq_batch_size, seq_len, *shape) -> (masked_batch_size, *shape)
        masked_next_obs = next_obs_seq[mask]
        masked_next_hidden_state = next_hidden_state_seq[mask]
        
        for mean_var, next_obs in zip(self._next_obs_feature_mean_var, masked_next_obs.items):
            mean_var.update(next_obs)
        self._next_hidden_state_feature_mean_var.update(masked_next_hidden_state)
        
        # normalize next observation and next hidden state
        normalized_next_obs_seq = next_obs_seq
        normalized_next_hidden_state_seq = next_hidden_state_seq
        normalized_next_obs_seq[mask] = self._normalize_next_obs(masked_next_obs)
        normalized_next_hidden_state_seq[mask] = self._normalize_next_hidden_state(masked_next_hidden_state)
        
        for _ in range(self._config.epoch):
            shuffled_seq_batch_idx = torch.randperm(seq_batch_size)
            for i in range(seq_batch_size // self._config.seq_mini_batch_size):
                # when sliced by sample_seq, (seq_batch_size,) -> (seq_mini_batch_size,)
                sample_seq_idx = shuffled_seq_batch_idx[self._config.seq_mini_batch_size * i : self._config.seq_mini_batch_size * (i + 1)]
                # when masked by m, (seq_mini_batch_size, seq_len,) -> (masked_batch_size,)
                sample_mask = mask[sample_seq_idx]
                
                # feed forward
                sample_policy_dist_seq, sample_ext_state_value_seq, sample_int_state_value_seq, _ = self._network.forward_actor_critic(
                    obs_seq[sample_seq_idx], 
                    seq_init_hidden_state[:, sample_seq_idx]
                )
                sample_predicted_feature, sample_target_feature = self._network.forward_rnd(
                    normalized_next_obs_seq[sample_seq_idx][sample_mask],
                    normalized_next_hidden_state_seq[sample_seq_idx][sample_mask].flatten(1, 2)
                )
                
                # compute actor loss
                sample_new_action_log_prob_seq = sample_policy_dist_seq.joint_log_prob(action_seq[sample_seq_idx])
                actor_loss = L.ppo_clipped_loss(
                    advantage_seq[sample_seq_idx][sample_mask],
                    old_action_log_prob_seq[sample_seq_idx][sample_mask],
                    sample_new_action_log_prob_seq[sample_mask],
                    self._config.epsilon_clip
                )
                entropy = sample_policy_dist_seq.joint_entropy()[sample_mask].mean()
                
                # compute critic loss
                ext_critic_loss = L.bellman_value_loss(
                    sample_ext_state_value_seq[sample_mask],
                    ext_target_state_value_seq[sample_seq_idx][sample_mask]
                )
                int_critic_loss = L.bellman_value_loss(
                    sample_int_state_value_seq[sample_mask], 
                    int_target_state_value_seq[sample_seq_idx][sample_mask]
                )
                critic_loss = ext_critic_loss + int_critic_loss
                
                # compute RND loss
                rnd_loss = L.rnd_loss(
                    sample_predicted_feature, 
                    sample_target_feature.detach(),
                    proportion=self._config.rnd_pred_exp_proportion
                )
                
                # train step
                loss = actor_loss + self._config.value_loss_coef * critic_loss - self._config.entropy_coef * entropy + rnd_loss
                self._trainer.step(loss, self.training_steps)
                self._tick_training_steps()
                
                # update log data
                self._actor_loss_mean.update(actor_loss.item())
                self._ext_critic_loss_mean.update(ext_critic_loss.item())
                self._int_critic_loss_mean.update(int_critic_loss.item())
                self._rnd_loss_mean.update(rnd_loss.item())

    def _compute_adv_target(self, exp_batch: RecurrentPPORNDExperience) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute advantage, extrinisc target state value, intrinsic target state value.
        
        Returns:
            advantage (Tensor): `(num_envs x n_steps, 1)`
            extrinisc target state value (Tensor): `(num_envs x n_steps, 1)`
            intrinsic target state value (Tensor): `(num_envs x n_steps, 1)`
        """
        # (num_envs, *obs_shape)
        final_next_obs = exp_batch.next_obs[-self._num_envs:]
        final_next_hidden_state = self._next_hidden_state
        
        # feed forward without gradient calculation
        with torch.no_grad():
            # (num_envs, 1, *obs_shape) because sequence length is 1
            _, final_ext_next_state_value_seq, final_int_next_state_value_seq, _ = self._network.forward_actor_critic(
                final_next_obs.transform(lambda o: o.unsqueeze(dim=1)),
                final_next_hidden_state
            )
        
        # (num_envs, 1, 1) -> (num_envs, 1)
        final_ext_next_state_value = final_ext_next_state_value_seq.squeeze_(dim=1)
        final_int_next_state_value = final_int_next_state_value_seq.squeeze_(dim=1)
        # (num_envs x (n_steps + 1), 1)
        entire_ext_state_value = torch.cat([exp_batch.ext_state_value, final_ext_next_state_value])
        entire_int_state_value = torch.cat([exp_batch.int_state_value, final_int_next_state_value])
                
        # (num_envs x k, 1) -> (num_envs, k, 1) -> (num_envs, k)
        b2e = lambda x: batch2perenv(x, self._num_envs).squeeze_(-1) 
        entire_ext_state_value = b2e(entire_ext_state_value)
        entire_int_state_value = b2e(entire_int_state_value)
        reward = b2e(exp_batch.ext_reward)
        terminated = b2e(exp_batch.terminated)
        int_reward = b2e(exp_batch.int_reward)
        
        # compute discounted intrinsic return
        discounted_int_return = torch.empty_like(int_reward)
        for t in range(self._config.n_steps):
            self._prev_discounted_int_return = int_reward[:, t] + self._config.int_gamma * self._prev_discounted_int_return
            discounted_int_return[:, t] = self._prev_discounted_int_return
        
        # update intrinic reward normalization parameters
        self._int_reward_mean_var.update(discounted_int_return)
        
        # normalize intinrisc reward
        int_reward /= torch.sqrt(self._int_reward_mean_var.variance).unsqueeze(dim=-1)
        self._int_reward_mean.update(int_reward.mean().item())
        
        # compute advantage (num_envs, n_steps) using GAE
        ext_advantage = L.gae(
            entire_ext_state_value,
            reward,
            terminated,
            self._config.ext_gamma,
            self._config.lam
        )
        int_advantage = L.gae(
            entire_int_state_value,
            int_reward,
            torch.zeros_like(terminated), # non-episodic
            self._config.int_gamma,
            self._config.lam
        )
        advantage = self._config.ext_adv_coef * ext_advantage + self._config.int_adv_coef * int_advantage
        
        # compute target state values (num_envs, n_steps)
        ext_target_state_value = ext_advantage + entire_ext_state_value[:, :-1]
        int_target_state_value = int_advantage + entire_int_state_value[:, :-1]
        
        # (num_envs, n_steps) -> (num_envs x n_steps, 1)
        e2b = lambda x: perenv2batch(x.unsqueeze_(-1))
        advantage = e2b(advantage)
        ext_target_state_value = e2b(ext_target_state_value)
        int_target_state_value = e2b(int_target_state_value)
        
        return advantage, ext_target_state_value, int_target_state_value
    
    def _compute_intrinsic_reward(self, next_obs: Observation, next_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward.
        
        Args:
            next_obs (Tensor): `(batch_size, *obs_shape)`
            next_hidden_state (Tensor): `(batch_size, D x num_layers, H)`
            
        Returns:
            intrinsic_reward (Tensor): `(batch_size, 1)`
        """
        with torch.no_grad():
            predicted_feature, target_feature = self._network.forward_rnd(next_obs, next_hidden_state.flatten(1, 2))
            intrinsic_reward = 0.5 * ((target_feature - predicted_feature)**2).sum(dim=1, keepdim=True)
            return intrinsic_reward
        
    def _normalize_next_obs(self, next_obs: Observation) -> Observation:
        """
        Normalize next observation. If `init_norm_steps` setting in the configuration is `None`, this method doesn't normalize it.
        """
        if self._config.init_norm_steps is None:
            return next_obs
        normalized_next_obs_tensors = []
        for mean_var, next_obs_tensor in zip(self._next_obs_feature_mean_var, next_obs.items):
            normalized_next_obs_tensor = ((next_obs_tensor - mean_var.mean) / torch.sqrt(mean_var.variance)).clamp(self._config.obs_norm_clip_range[0], self._config.obs_norm_clip_range[1])
            normalized_next_obs_tensors.append(normalized_next_obs_tensor)
        return Observation(tuple(normalized_next_obs_tensors))

    def _normalize_next_hidden_state(self, next_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Normalize next hidden state. If `init_norm_steps` setting in the configuration is `None`, this method doesn't normalize it.
        """
        if self._config.init_norm_steps is None:
            return next_hidden_state
        hidden_state_feature_mean = self._next_hidden_state_feature_mean_var.mean
        hidden_state_feature_std = torch.sqrt(self._next_hidden_state_feature_mean_var.variance)
        normalized_next_hidden_state = (next_hidden_state - hidden_state_feature_mean) / hidden_state_feature_std
        return normalized_next_hidden_state.clip(self._config.hidden_state_norm_clip_range[0], self._config.hidden_state_norm_clip_range[1])
    
    @property
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/Actor Loss", "Network/Extrinsic Critic Loss", "Network/Intrinsic Critic Loss", "Network/RND Loss", "RND/Intrinsic Reward")
    
    @property
    def log_data(self) -> dict[str, tuple]:
        ld = super().log_data
        if self._actor_loss_mean.count > 0:
            ld["Network/Actor Loss"] = (self._actor_loss_mean.mean, self.training_steps)
            ld["Network/Extrinsic Critic Loss"] = (self._ext_critic_loss_mean.mean, self.training_steps)
            ld["Network/Intrinsic Critic Loss"] = (self._int_critic_loss_mean.mean, self.training_steps)
            ld["Network/RND Loss"] = (self._rnd_loss_mean.mean, self.training_steps)
            self._actor_loss_mean.reset()
            self._ext_critic_loss_mean.reset()
            self._int_critic_loss_mean.reset()
            self._rnd_loss_mean.reset()
        if self._int_reward_mean.count > 0:
            ld["RND/Intrinsic Reward"] = (self._int_reward_mean.mean, self.training_steps)
            self._int_reward_mean.reset()
        return ld
