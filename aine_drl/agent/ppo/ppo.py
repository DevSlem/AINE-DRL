from __future__ import annotations

import torch

import aine_drl.rl_loss as L
import aine_drl.util as util
from aine_drl.agent.agent import Agent, BehaviorType
from aine_drl.agent.ppo.config import PPOConfig
from aine_drl.agent.ppo.net import PPOSharedNetwork
from aine_drl.agent.ppo.trajectory import PPOExperience, PPOTrajectory
from aine_drl.exp import Action, Experience, Observation
from aine_drl.net import NetworkTypeError, Trainer
from aine_drl.util.func import batch2perenv, perenv2batch


class PPO(Agent):
    """
    Proximal Policy Optimization (PPO). 
    
    Paper: https://arxiv.org/abs/1707.06347
    """
    def __init__(
        self, 
        config: PPOConfig,
        network: PPOSharedNetwork,
        trainer: Trainer,
        num_envs: int,
        behavior_type: BehaviorType = BehaviorType.TRAIN,
    ) -> None:        
        if not isinstance(network, PPOSharedNetwork):
            raise NetworkTypeError(PPOSharedNetwork)
        
        super().__init__(num_envs, network, config.device, behavior_type)
        
        self._config = config
        self._network = network
        self._trainer = trainer
        self._trajectory = PPOTrajectory(self._config.n_steps)
        
        self._action_log_prob: torch.Tensor = None # type: ignore
        self._state_value: torch.Tensor = None # type: ignore
        
        self._actor_loss_mean = util.IncrementalMean()
        self._critic_loss_mean = util.IncrementalMean()
        
    @property
    def name(self) -> str:
        return "PPO"
    
    @property
    def config_dict(self) -> dict:
        return self._config.__dict__
    
    def _update_train(self, exp: Experience):
        # add the experience
        self._trajectory.add(PPOExperience(
            **exp.__dict__,
            action_log_prob=self._action_log_prob,
            state_value=self._state_value
        ))
        
        if self._trajectory.reached_n_steps:
            self._train()
    
    def _update_inference(self, _: Experience):
        pass
    
    @torch.no_grad()
    def _select_action_train(self, obs: Observation) -> Action:
        # feed forward 
        policy_dist, v_pred = self._network.forward(obs)
        
        # action sampling
        action = policy_dist.sample()
        
        # store data
        self._action_log_prob = policy_dist.joint_log_prob(action)
        self._state_value = v_pred
        
        return action
    
    @torch.no_grad()
    def _select_action_inference(self, obs: Observation) -> Action:
        policy_dist, _ = self._network.forward(obs)
        return policy_dist.sample()
            
    def _train(self):
        exp_batch = self._trajectory.sample()
        batch_size = self.num_envs * self._config.n_steps
        
        old_action_log_prob = exp_batch.action_log_prob
        advantage, target_state_value = self._compute_adv_target(exp_batch)
        
        for _ in range(self._config.epoch):
            shuffled_batch_idx = torch.randperm(batch_size)
            for i in range(batch_size // self._config.mini_batch_size):
                sample_batch_idx = shuffled_batch_idx[self._config.mini_batch_size * i : self._config.mini_batch_size * (i + 1)]
                
                # feed forward
                policy_dist, state_value = self._network.forward(exp_batch.obs[sample_batch_idx])
                
                # compute actor loss
                new_action_log_prob = policy_dist.joint_log_prob(exp_batch.action[sample_batch_idx])
                normalized_advantage = self._normalize(advantage[sample_batch_idx]) if self._config.advantage_normalization else advantage[sample_batch_idx]
                actor_loss = L.ppo_clipped_loss(
                    normalized_advantage,
                    old_action_log_prob[sample_batch_idx],
                    new_action_log_prob,
                    self._config.epsilon_clip
                )
                entropy = policy_dist.joint_entropy().mean()
                
                # compute critic loss
                critic_loss = L.bellman_value_loss(state_value, target_state_value[sample_batch_idx])
                
                # train step
                loss = actor_loss + self._config.value_loss_coef * critic_loss - self._config.entropy_coef * entropy
                self._trainer.step(loss, self.training_steps)
                self._tick_training_steps()
                
                # log data
                self._actor_loss_mean.update(actor_loss.item())
                self._critic_loss_mean.update(critic_loss.item())

        
    def _compute_adv_target(self, exp_batch: PPOExperience) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantage, v_target. `batch_size` is `num_evns` x `n_steps`.

        Args:
            exp_batch (PPOExperience): experience batch

        Returns:
            tuple[Tensor, Tensor]: advantage, v_target whose each shape is `(batch_size, 1)`
        """
        with torch.no_grad():
            final_next_obs = exp_batch.next_obs[-self._num_envs:]
            _, final_next_v_pred = self._network.forward(final_next_obs)
        
        entire_state_value = torch.cat([exp_batch.state_value, final_next_v_pred])
        
        # (num_envs * n + 1, 1) -> (num_envs, n, 1) -> (num_envs, n)
        b2e = lambda x: batch2perenv(x, self._num_envs).squeeze_(-1)
        entire_state_value = b2e(entire_state_value)
        reward = b2e(exp_batch.reward)
        terminated = b2e(exp_batch.terminated)
        
        # compute advantage using GAE
        advantage = L.gae(
            entire_state_value,
            reward,
            terminated,
            self._config.gamma,
            self._config.lam
        )
        
        # compute target state value
        target_state_value = advantage + entire_state_value[:, :-1]
        
        e2b = lambda x: perenv2batch(x.unsqueeze_(-1))
        advantage = e2b(advantage)
        target_state_value = e2b(target_state_value)
        
        return advantage, target_state_value


    def _normalize(self, x: torch.Tensor, mask: bool | torch.Tensor = True) -> torch.Tensor:
        return (x - x[mask].mean()) / (x[mask].std() + 1e-8)

    @property
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/Actor Loss", "Network/Critic Loss")
    
    @property
    def log_data(self) -> dict[str, tuple]:
        ld = super().log_data
        if self._actor_loss_mean.count > 0:
            ld["Network/Actor Loss"] = (self._actor_loss_mean.mean, self.training_steps)
            ld["Network/Critic Loss"] = (self._critic_loss_mean.mean, self.training_steps)
            self._actor_loss_mean.reset()
            self._critic_loss_mean.reset()
        return ld
