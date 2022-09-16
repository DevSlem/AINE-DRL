from aine_drl.agent.agent import Agent
from aine_drl.drl_util.clock import Clock
from aine_drl.policy.policy import Policy
from aine_drl.trajectory.trajectory import Trajectory
from typing import NamedTuple, Tuple, Union
import aine_drl.util as util
import aine_drl.drl_util as drl_util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class ActorCriticNetSpec(NamedTuple):
    policy_net: nn.Module
    value_net: nn.Module
    policy_optimizer: optim.Optimizer
    value_optimizer: optim.Optimizer
    policy_lr_scheduler: _LRScheduler = None # Defaults to not scheduling
    value_lr_scheduler: _LRScheduler = None # Defaults to not scheduling
    
class ActorCriticSharedNetSpec(NamedTuple):
    policy_net: nn.Module
    value_net: nn.Module
    optimizer: optim.Optimizer # optimizer must be able to update policy_net, value_net, shared_net at once.
    value_loss_coef: float = 0.5
    lr_scheduler: _LRScheduler = None # Defaults to not scheduling

class A2C(Agent):
    """
    Advantage Actor Critic (A2C).
    """
    def __init__(self,
                 net_spec: Union[ActorCriticNetSpec, ActorCriticSharedNetSpec],
                 policy: Policy, 
                 trajectory: Trajectory, 
                 clock: Clock,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 summary_freq: int = 1000) -> None:
        """
        Advantage Actor Critic (A2C).

        Args:
            net_spec (ActorCriticNetSpec | ActorCriticSharedNetSpec): network spec
            policy (Policy): policy for action selection
            trajectory (Trajectory): on-policy trajectory for sampling
            clock (Clock): time step checker
            gamma (float, optional): discount factor. Defaults to 0.99.
            lam (float, optional): lambda which controls the GAE balanace between bias and variance. Defaults to 0.95.
            summary_freq (int, optional): summary frequency. Defaults to 1000.
        """
        super().__init__(policy, trajectory, clock, summary_freq)
        self.net_spec = net_spec
        self.shared_net = type(net_spec) == ActorCriticSharedNetSpec
        self.gamma = gamma
        self.lam = lam
        self.num_envs = clock.num_envs
        self.value_loss_func = nn.MSELoss()
        self.device = util.get_model_device(net_spec.policy_net)
        self.action_log_probs = []
        self.policy_losses = []
        self.value_losses = []        
        
    def select_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        pdparam = self.net_spec.policy_net(state.to(device=self.device))
        dist = self.policy.get_policy_distribution(pdparam)
        action = dist.sample()
        # save action log probability to compute policy loss
        self.action_log_probs.append(dist.log_prob(action))
        return action
        
    def train(self):
        # compute loss
        batch = self.trajectory.sample()
        states, actions, next_states, rewards, terminateds = batch.to_tensor(self.device)
        v_preds = self.net_spec.value_net(states)
        advantages, v_targets = self.compute_advantage_v_target(v_preds, next_states, rewards, terminateds)
        policy_loss = self.compute_policy_loss(advantages) # actor
        value_loss = self.compute_value_loss(v_preds, v_targets) # critic
        # update policy, value network
        if self.shared_net:
            loss = policy_loss + self.net_spec.value_loss_coef * value_loss
            util.train_step(loss, self.net_spec.optimizer, self.net_spec.lr_scheduler, self.clock.training_step)
        else:
            util.train_step(policy_loss, self.net_spec.policy_optimizer, self.net_spec.policy_lr_scheduler, self.clock.training_step)
            util.train_step(value_loss, self.net_spec.value_optimizer, self.net_spec.value_lr_scheduler, self.clock.training_step)
        self.clock.tick_training_step()
        # update data
        self.action_log_probs.clear()
        self.policy_losses.append(policy_loss.detach().cpu().item())
        self.value_losses.append(value_loss.detach().cpu().item())
    
    def compute_advantage_v_target(self, 
                                   v_preds: torch.Tensor,
                                   next_states: torch.Tensor, 
                                   rewards: torch.Tensor, 
                                   terminateds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes advantages for the actor and v_targets for the critic.
        """
        with torch.no_grad():
            final_next_v_pred = self.net_spec.value_net(next_states[-self.num_envs:])
        v_preds_no_grad = v_preds.detach()
        v_preds_all = torch.cat([v_preds_no_grad, final_next_v_pred]).squeeze_(-1)
        gae = drl_util.calc_gae(
            util.vector_env_pack(rewards, self.num_envs),
            util.vector_env_pack(terminateds, self.num_envs),
            util.vector_env_pack(v_preds_all, self.num_envs),
            self.gamma,
            self.lam
        )
        advantages = util.vector_env_unpack(gae)
        v_target = advantages.unsqueeze_(-1) + v_preds_no_grad
        return advantages, v_target
    
    def compute_policy_loss(self, advantages: torch.Tensor):
        """
        Actor
        """
        log_probs = torch.cat(self.action_log_probs).unsqueeze_(-1)
        policy_loss =  -(advantages * log_probs).sum()
        return policy_loss
    
    def compute_value_loss(self, v_preds: torch.Tensor, v_targets: torch.Tensor) -> torch.Tensor:
        """
        Critic
        """
        return self.value_loss_func(v_targets, v_preds)
    
    def log_data(self, time_step: int):
        super().log_data(time_step)
        if len(self.policy_losses) > 0:
            util.log_data("Network/Policy Loss", np.mean(self.policy_losses), time_step)
            util.log_data("Network/Value Loss", np.mean(self.value_losses), time_step)
            self.policy_losses.clear()
            self.value_losses.clear()
        if self.shared_net:
            util.log_lr_scheduler(self.net_spec.lr_scheduler, time_step)
        else:
            util.log_lr_scheduler(self.net_spec.policy_lr_scheduler, time_step, "Policy Network Learning Rate")
            util.log_lr_scheduler(self.net_spec.value_lr_scheduler, time_step, "Value Network Learning Rate")
