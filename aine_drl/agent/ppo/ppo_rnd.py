import torch

import aine_drl.rl_loss as L
import aine_drl.util as util
from aine_drl.agent.agent import Agent, BehaviorType
from aine_drl.agent.ppo.config import PPORNDConfig
from aine_drl.agent.ppo.net import PPORNDNetwork
from aine_drl.agent.ppo.trajectory import PPORNDExperience, PPORNDTrajectory
from aine_drl.exp import Action, Experience, Observation
from aine_drl.net import NetworkTypeError, Trainer
from aine_drl.util.func import batch2perenv, perenv2batch


class PPORND(Agent):
    """
    Proximal Policy Optimization (PPO) with Random Network Distillation (RND).
    
    PPO paper: https://arxiv.org/abs/1707.06347 \\
    RND paper: https://arxiv.org/abs/1810.12894
    """
    def __init__(
        self, 
        config: PPORNDConfig,
        network: PPORNDNetwork, 
        trainer: Trainer,
        num_envs: int, 
        behavior_type: BehaviorType = BehaviorType.TRAIN
    ) -> None:
        if not isinstance(network, PPORNDNetwork):
            raise NetworkTypeError(PPORNDNetwork)
        
        super().__init__(num_envs, network, config.device, behavior_type)
        
        self._config = config
        self._network = network
        self._trainer = trainer
        self._trajectory = PPORNDTrajectory(self._config.n_steps)
        
        self._action_log_prob: torch.Tensor = None # type: ignore
        self._ext_state_value: torch.Tensor = None # type: ignore
        self._int_state_value: torch.Tensor = None # type: ignore
        
        self._prev_discounted_int_return = 0.0
        # compute intrinic reward normalization parameters of each env along time steps
        self._int_reward_mean_var = util.IncrementalMeanVarianceFromBatch(dim=1, device=self.device) 
        # compute normalization parameters of each feature of next observation along batches
        self._next_obs_feature_mean_var: tuple[util.IncrementalMeanVarianceFromBatch] = tuple()
        self._current_init_norm_steps = 0
        
        self._actor_loss_mean = util.IncrementalMean()
        self._ext_critic_loss_mean = util.IncrementalMean()
        self._int_critic_loss_mean = util.IncrementalMean()
        self._rnd_loss_mean = util.IncrementalMean()
        self._int_reward_mean = util.IncrementalMean()
    
    @property
    def name(self) -> str:
        return "PPO RND"
    
    def _update_train(self, exp: Experience):
        # initialize normalization parameters
        if (self._config.init_norm_steps is not None) and (self._current_init_norm_steps < self._config.init_norm_steps):
            self._current_init_norm_steps += 1
            
            if len(self._next_obs_feature_mean_var) == 0:
                self._next_obs_feature_mean_var = tuple(
                    util.IncrementalMeanVarianceFromBatch(dim=0, device=self.device) for _ in range(exp.next_obs.num_items)
                )
            self._update_next_obs_norm_params(exp.next_obs)
            
        # compute intrinsic reward
        normalized_next_obs = self._normalize_next_obs(exp.next_obs)
        int_reward = self._compute_intrinsic_reward(normalized_next_obs)
        
        # add one experience
        self._trajectory.add(PPORNDExperience(
            obs=exp.obs,
            action=exp.action,
            next_obs=exp.next_obs,
            ext_reward=exp.reward,
            int_reward=int_reward,
            terminated=exp.terminated,
            action_log_prob=self._action_log_prob,
            ext_state_value=self._ext_state_value,
            int_state_value=self._int_state_value,
        ))
        
        if self._trajectory.reached_n_steps:
            self._train()
    
    def _update_inference(self, exp: Experience):
        pass
    
    @torch.no_grad()
    def _select_action_train(self, obs: Observation) -> Action:
        # feed forward
        policy_dist, ext_state_value, int_state_value = self._network.forward_actor_critic(obs)
        
        # action sampling
        action = policy_dist.sample()
        
        self._action_log_prob = policy_dist.log_prob(action)
        self._ext_state_value = ext_state_value
        self._int_state_value = int_state_value
        
        return action
    
    @torch.no_grad()
    def _select_action_inference(self, obs: Observation) -> Action:
        policy_dist, _, _ = self._network.forward_actor_critic(obs)
        return policy_dist.sample()
    
    def _train(self):
        exp_batch = self._trajectory.sample()
        batch_size = self.num_envs * self._config.n_steps

        # compute advantage and target state value
        advantage, ext_target_state_value, int_target_state_value = self._compute_adv_target(exp_batch)
        old_action_log_prob = exp_batch.action_log_prob
        
        # update next observation and next hidden state normalization parameters
        self._update_next_obs_norm_params(exp_batch.next_obs)
        
        # normalize next observation
        normalized_next_obs = self._normalize_next_obs(exp_batch.next_obs)
        
        for _ in range(self._config.epoch):
            shuffled_batch_idx = torch.randperm(batch_size)
            for i in range(batch_size // self._config.mini_batch_size):
                sample_batch_idx = shuffled_batch_idx[i * self._config.mini_batch_size : (i + 1) * self._config.mini_batch_size]
                
                # feed forward
                sample_policy_dist, sample_ext_state_value, sample_int_state_value = self._network.forward_actor_critic(exp_batch.obs[sample_batch_idx])
                sample_predicted_feature, sample_target_feature = self._network.forward_rnd(normalized_next_obs[sample_batch_idx])
                
                # compute actor loss
                sample_new_action_log_prob = sample_policy_dist.joint_log_prob(exp_batch.action[sample_batch_idx])
                actor_loss = L.ppo_clipped_loss(
                    advantage[sample_batch_idx],
                    old_action_log_prob[sample_batch_idx],
                    sample_new_action_log_prob,
                    self._config.epsilon_clip
                )
                entropy = sample_policy_dist.joint_entropy().mean()
                
                # compute critic loss
                ext_critic_loss = L.bellman_value_loss(
                    sample_ext_state_value,
                    ext_target_state_value[sample_batch_idx]
                )
                int_critic_loss = L.bellman_value_loss(
                    sample_int_state_value,
                    int_target_state_value[sample_batch_idx]
                )
                critic_loss = ext_critic_loss + int_critic_loss
                
                # compute RND loss
                rnd_loss = L.rnd_loss(
                    sample_predicted_feature,
                    sample_target_feature.detach(),
                    proportion=self._config.rnd_pred_exp_proportion
                )
                
                # train step
                loss = actor_loss + self._config.value_loss_coef * critic_loss + rnd_loss - self._config.entropy_coef * entropy
                self._trainer.step(loss, self.training_steps)
                self._tick_training_steps()
                
                # update log data
                self._actor_loss_mean.update(actor_loss.item())
                self._ext_critic_loss_mean.update(ext_critic_loss.item())
                self._int_critic_loss_mean.update(int_critic_loss.item())
                self._rnd_loss_mean.update(rnd_loss.item())
        
    def _compute_adv_target(self, exp_batch: PPORNDExperience) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute advantage, extrinisc target state value, intrinsic target state value.
        
        Returns:
            advantage (Tensor): `(num_envs x n_steps, 1)`
            extrinisc target state value (Tensor): `(num_envs x n_steps, 1)`
            intrinsic target state value (Tensor): `(num_envs x n_steps, 1)`
        """
        # (num_envs, *obs_shape)
        final_next_obs = exp_batch.next_obs[-self._num_envs:]
        
        # feed forward without gradient calculation
        with torch.no_grad():
            _, final_ext_next_state_value, final_int_next_state_value = self._network.forward_actor_critic(final_next_obs)
        
        # (num_envs x (n_steps + 1), 1)
        entire_ext_state_value = torch.cat([exp_batch.ext_state_value, final_ext_next_state_value])
        entire_int_state_value = torch.cat([exp_batch.int_state_value, final_int_next_state_value])
                
        # (num_envs x k, 1) -> (num_envs, k, 1) -> (num_envs, k)
        b2e = lambda x: batch2perenv(x, self._num_envs).squeeze_(-1) 
        entire_ext_state_value = b2e(entire_ext_state_value)
        entire_int_state_value = b2e(entire_int_state_value)
        ext_reward = b2e(exp_batch.ext_reward)
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
            ext_reward,
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
    
    def _update_next_obs_norm_params(self, next_obs: Observation):
        for mean_var, obs in zip(self._next_obs_feature_mean_var, next_obs.items):
            mean_var.update(obs)
            
    def _compute_intrinsic_reward(self, next_obs: Observation) -> torch.Tensor:
        """
        Compute intrinsic reward.
        
        Args:
            next_obs (Tensor): `(batch_size, *obs_shape)`
            
        Returns:
            intrinsic_reward (Tensor): `(batch_size, 1)`
        """
        with torch.no_grad():
            predicted_feature, target_feature = self._network.forward_rnd(next_obs)
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
    