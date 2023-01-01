from typing import Dict, NamedTuple, Optional, Tuple
from aine_drl.agent.agent import Agent
from aine_drl.experience import ActionTensor, Experience, ExperienceBatchTensor
from aine_drl.network import SACNetwork
from aine_drl.policy.policy import ActionType, Policy
from aine_drl.trajectory.experience_replay import ExperienceReplay
import aine_drl.drl_util as drl_util
import aine_drl.util as util
import torch
import torch.nn.functional as F
import copy

class SACConfig(NamedTuple):
    training_freq: int
    batch_size: int
    capacity: int
    epoch: int
    gamma: float = 0.99
    polyak_ratio: float = 0.005
    alpha: float = 0.2
    grad_clip_max_norm: Optional[float] = None

class SAC(Agent):
    def __init__(self, 
                 config: SACConfig, 
                 network: SACNetwork, 
                 policy: Policy, 
                 num_envs: int) -> None:
        raise NotImplementedError("It's not implemented yet. I've not been able to find the reason why it fails training.")
        
        if policy.action_type is not ActionType.CONTINUOUS:
            raise TypeError(f"The policy must be continuous action policy, but \"{type(policy).__name__}\" is \"{policy.action_type}\".")
        
        super().__init__(network, policy, num_envs)
        
        self.config = config
        self.network = network
        self.v_target_net = copy.deepcopy(network.v_net)
        self.trajectory = ExperienceReplay(
            self.config.training_freq,
            self.config.batch_size,
            self.config.capacity,
            self.num_envs,
            online=True
        )
        
        self.average_soft_value_loss = util.IncrementalAverage()
        self.average_td_loss1 = util.IncrementalAverage()
        self.average_td_loss2 = util.IncrementalAverage()
        self.average_policy_kl_loss = util.IncrementalAverage()
        
    def select_action_train(self, obs: torch.Tensor) -> ActionTensor:
        with torch.no_grad():
            return self._sample_action_tensor(obs)
    
    def select_action_inference(self, obs: torch.Tensor) -> ActionTensor:
        return self._sample_action_tensor(obs)
        
    def update(self, experience: Experience):
        super().update(experience)
        
        self.trajectory.add(experience)
        
        if self.trajectory.can_sample:
            self.train()
        
    def train(self):
        for _ in range(self.config.epoch):
            exp_batch = self.trajectory.sample(device=self.device)
            self._update_parameters(exp_batch)
            
            self.clock.tick_training_step()
            
    @property
    def log_keys(self) -> Tuple[str, ...]:
        return super().log_keys + ("Network/Soft Value Loss", "Network/TD Loss (1)", "Network/TD Loss (2)", "Network/Policy KL Loss")
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        ld = super().log_data
        if self.average_soft_value_loss.count > 0:
            ld["Network/Soft Value Loss"] = (self.average_soft_value_loss.average, self.clock.training_step)
            ld["Network/TD Loss (1)"] = (self.average_td_loss1.average, self.clock.training_step)
            ld["Network/TD Loss (2)"] = (self.average_td_loss2.average, self.clock.training_step)
            ld["Network/Policy KL Loss"] = (self.average_policy_kl_loss.average, self.clock.training_step)
            self.average_soft_value_loss.reset()
            self.average_td_loss1.reset()
            self.average_td_loss2.reset()
            self.average_policy_kl_loss.reset()
        return ld
    
    def _sample_action_tensor(self, obs: torch.Tensor) -> ActionTensor:
        pdparam = self.network.actor.forward(obs)
        dist = self.policy.get_policy_distribution(pdparam)
        return dist.sample()
    
    def _update_parameters(self, exp_batch: ExperienceBatchTensor):
        # === objective function J_Q: 0.5 * (Q(s,a) - q_hat(s,a))^2 ===
        with torch.no_grad():
            # compute q_hat
            not_terminated = 1.0 - exp_batch.terminated
            next_v_target = self.v_target_net.forward(exp_batch.next_obs)
            q_hat = exp_batch.reward + not_terminated * self.config.gamma * next_v_target
            
        q_pred1, q_pred2 = self._estimate_two_continuous_q_value(exp_batch.obs, exp_batch.action)
            
        # compute objective function: td loss
        td_loss1 = F.mse_loss(q_pred1, q_hat)
        td_loss2 = F.mse_loss(q_pred2, q_hat)
        
        # gradient step
        self.network.q_net1.train_step(td_loss1, self.config.grad_clip_max_norm, self.clock.training_step)
        self.network.q_net2.train_step(td_loss2, self.config.grad_clip_max_norm, self.clock.training_step)
        
        # record
        self.average_td_loss1.update(td_loss1.item())
        self.average_td_loss2.update(td_loss2.item())
        
        # === objective function J_pi: log(pi(a|s)) - Q(s,a) ===
        # sample actions
        pdparam = self.network.actor.forward(exp_batch.obs)
        dist = self.policy.get_policy_distribution(pdparam)
        # reparameterized trick must be used
        sampled_action = dist.sample()
        action_log_prob = dist.log_prob(sampled_action)
        
        # estimate Q(s,a)
        q_pred1, q_pred2 = self._estimate_two_continuous_q_value(exp_batch.obs, sampled_action)
        min_q_pred = torch.min(q_pred1, q_pred2)
        
        # compute objective function: policy kl loss
        policy_kl_loss = (self.config.alpha * action_log_prob - min_q_pred).mean()
        
        # gradient step
        self.network.actor.train_step(policy_kl_loss, self.config.grad_clip_max_norm, self.clock.training_step)
        
        # record
        self.average_policy_kl_loss.update(policy_kl_loss.item())
        
        # === objective function J_V: 0.5 * (V(s) - Q(s,a) + log(pi(a|s)))^2 ===
        with torch.no_grad():
            v_target = min_q_pred - (self.config.alpha * action_log_prob)
        
        # estimate V(s), its gradient only flows
        v_pred = self.network.v_net.forward(exp_batch.obs)
        
        # compute objective function: soft value loss
        soft_value_loss = F.mse_loss(v_pred, v_target)
        
        # gradient step
        self.network.v_net.train_step(soft_value_loss, self.config.grad_clip_max_norm, self.clock.training_step)
        
        # record
        self.average_soft_value_loss.update(soft_value_loss.item())
        
        # === target network update ===
        self._update_v_target_net()
        
        
    def _estimate_two_continuous_q_value(self, obs: torch.Tensor, action: ActionTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_pred1 = self.network.q_net1.forward(obs, action.continuous_action)
        q_pred2 = self.network.q_net2.forward(obs, action.continuous_action)
        return q_pred1, q_pred2
    
    def _update_v_target_net(self):
        drl_util.polyak_update(self.network.v_net, self.v_target_net, self.config.polyak_ratio)
    