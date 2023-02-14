from typing import Dict, Tuple
from aine_drl.agent import Agent
from aine_drl.experience import ActionTensor, Experience
from aine_drl.network import NetworkTypeError
from aine_drl.policy.policy import Policy
from .config import PPOConfig
from .net import PPOSharedNetwork
from .ppo_trajectory import PPOExperienceBatch, PPOTrajectory
import aine_drl.drl_util as drl_util
import aine_drl.util as util
import torch
import torch.nn.functional as F

class PPO(Agent):
    """
    Proximal Policy Optimization (PPO). See details in https://arxiv.org/abs/1707.06347.

    Args:
        config (PPOConfig): PPO configuration
        network (ActorCriticSharedNetwork): standard actor critic network
        policy (Policy): policy
        num_envs (int): number of environments
    """
    def __init__(self, 
                 config: PPOConfig,
                 network: PPOSharedNetwork,
                 policy: Policy,
                 num_envs: int) -> None:        
        if not isinstance(network, PPOSharedNetwork):
            raise NetworkTypeError(PPOSharedNetwork)
        
        super().__init__(network, policy, num_envs)
        
        self.config = config
        self.network = network
        self.trajectory = PPOTrajectory(self.config.training_freq)
        
        self.current_action_log_prob = None
        self.v_pred = None
        
        self.actor_average_loss = util.IncrementalAverage()
        self.critic_average_loss = util.IncrementalAverage()
        
    def update(self, experience: Experience):
        super().update(experience)
        
        # add the experience
        self.trajectory.add(
            experience,
            self.current_action_log_prob,
            self.v_pred,
        )
        
        # if training frequency is reached, start training
        if self.trajectory.count == self.config.training_freq:
            self.train()
            
    def select_action_train(self, obs: torch.Tensor) -> ActionTensor:
        with torch.no_grad():
            # feed forward 
            pdparam, v_pred = self.network.forward(obs)
            
            # action sampling
            dist = self.policy.get_policy_distribution(pdparam)
            action = dist.sample()
            
            # store data
            self.current_action_log_prob = dist.joint_log_prob(action).cpu()
            self.v_pred = v_pred.cpu()
            
            return action
    
    def select_action_inference(self, obs: torch.Tensor) -> ActionTensor:
        pdparam, _ = self.network.forward(obs)
        dist = self.policy.get_policy_distribution(pdparam)
        return dist.sample()
            
    def train(self):
        exp_batch = self.trajectory.sample(self.device)
        batch_size = len(exp_batch.obs)
        
        old_action_log_prob = exp_batch.action_log_prob
        advantage, v_target = self.compute_adavantage_v_target(exp_batch)
        
        for _ in range(self.config.epoch):
            sample_idxes = torch.randperm(batch_size)
            for i in range(batch_size // self.config.mini_batch_size):
                sample_idx = sample_idxes[self.config.mini_batch_size * i : self.config.mini_batch_size * (i + 1)]
                
                # feed forward
                pdparam, v_pred = self.network.forward(exp_batch.obs[sample_idx])
                
                # compute actor loss
                dist = self.policy.get_policy_distribution(pdparam)
                new_action_log_prob = dist.joint_log_prob(exp_batch.action[sample_idx])
                adv = drl_util.normalize(advantage[sample_idx]) if self.config.advantage_normalization else advantage[sample_idx]
                actor_loss = PPO.compute_actor_loss(
                    adv,
                    old_action_log_prob[sample_idx],
                    new_action_log_prob,
                    self.config.epsilon_clip
                )
                entropy = dist.joint_entropy().mean()
                
                # compute critic loss
                critic_loss = PPO.compute_critic_loss(v_pred, v_target[sample_idx])
                
                # train step
                loss = actor_loss + self.config.value_loss_coef * critic_loss - self.config.entropy_coef * entropy
                self.network.train_step(loss, self.config.grad_clip_max_norm, self.clock.training_step)
                
                self.clock.tick_training_step()
                
                # log data
                self.actor_average_loss.update(actor_loss.item())
                self.critic_average_loss.update(critic_loss.item())

        
    def compute_adavantage_v_target(self, exp_batch: PPOExperienceBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantage, v_target. `batch_size` is `num_evns` x `n_steps`.

        Args:
            exp_batch (PPOExperienceBatch): experience batch

        Returns:
            Tuple[Tensor, Tensor]: advantage, v_target whose each shape is `(batch_size, 1)`
        """
        with torch.no_grad():
            final_next_obs = exp_batch.next_obs[-self.num_envs:]
            _, final_next_v_pred = self.network.forward(final_next_obs)
        
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
    
    @staticmethod
    def compute_actor_loss(advantage: torch.Tensor, 
                           old_action_log_prob: torch.Tensor,
                           new_action_log_prob: torch.Tensor,
                           epsilon_clip: float = 0.2) -> torch.Tensor:
        """
        Compute actor loss using PPO. It uses mean loss not sum loss.

        Args:
            advantage (Tensor): whose shape is `(batch_size, 1)`
            old_action_log_prob (Tensor): log(pi_theta_old) whose gradient never flows and shape is `(batch_size, 1)`
            new_action_log_prob (Tensor): log(pi_theta_new) whose gradient flows and shape is `(batch_size, 1)`
            epsilon_clip (float, optional): clipped range is [1 - epsilon, 1 + epsilon]. Defaults to 0.2.

        Returns:
            Tensor: PPO actor loss
        """
        assert not old_action_log_prob.requires_grad, "gradients of new_action_log_prob only flows."
        # pi_theta / pi_theta_old
        ratios = torch.exp(new_action_log_prob - old_action_log_prob)
        
        # surrogate loss
        sur1 = ratios * advantage
        sur2 = torch.clamp(ratios, 1 - epsilon_clip, 1 + epsilon_clip) * advantage
        
        # compute actor loss
        loss = -torch.min(sur1, sur2).mean()
        return loss
    
    @staticmethod
    def compute_critic_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        """
        Compute critic loss using MSE.

        Args:
            v_pred (Tensor): predicted state value whose gradient flows and shape is `(batch_size, 1)`
            v_target (Tensor): target state value whose gradient never flows and shape is `(batch_size, 1)`

        Returns:
            Tensor: critic loss
        """
        return F.mse_loss(v_target, v_pred)

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
