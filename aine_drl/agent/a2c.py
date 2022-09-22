from typing import Tuple, Union
from dataclasses import dataclass
from aine_drl.drl_util import NetSpec
from aine_drl.agent.agent import Agent
from aine_drl.drl_util.clock import Clock
from aine_drl.policy.policy import Policy
from aine_drl.trajectory.trajectory import Trajectory
from aine_drl.util import logger
import aine_drl.util as util
import aine_drl.drl_util as drl_util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

@dataclass
class ActorCriticNetSpec(NetSpec):
    policy_net: nn.Module
    value_net: nn.Module
    policy_optimizer: optim.Optimizer
    value_optimizer: optim.Optimizer
    policy_lr_scheduler: _LRScheduler = None # Defaults to not scheduling
    value_lr_scheduler: _LRScheduler = None # Defaults to not scheduling
    grad_clip_max_norm: Union[float, None] = None
    
    def train_step(self, policy_loss: torch.Tensor, value_loss: torch.Tensor, current_epoch: int):
        util.train_step(
            policy_loss, 
            self.policy_optimizer, 
            self.policy_lr_scheduler, 
            self.grad_clip_max_norm, 
            current_epoch
        )
        util.train_step(
            value_loss, 
            self.value_optimizer, 
            self.value_lr_scheduler, 
            self.grad_clip_max_norm, 
            current_epoch
        )
        
    @property
    def state_dict(self) -> dict:
        sd = {
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "policy_lr_scheduler": self.policy_lr_scheduler.state_dict() if self.policy_lr_scheduler is not None else None,
            "value_lr_scheduler": self.value_lr_scheduler.state_dict() if self.value_lr_scheduler is not None else None,
            "grad_clip_max_norm": self.grad_clip_max_norm
        }
        return sd
        
    def load_state_dict(self, state_dict: dict):
        self.policy_net.load_state_dict(state_dict["policy_net"])
        self.value_net.load_state_dict(state_dict["value_net"])
        self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])
        self.value_optimizer.load_state_dict(state_dict["value_optimizer"])
        if self.policy_lr_scheduler is not None:
            self.policy_lr_scheduler.load_state_dict(state_dict["policy_lr_scheduler"])
        if self.value_lr_scheduler is not None:
            self.value_lr_scheduler.load_state_dict(state_dict["value_lr_scheduler"])
        self.grad_clip_max_norm = state_dict["grad_clip_max_norm"]
    
@dataclass
class ActorCriticSharedNetSpec(NetSpec):
    policy_net: nn.Module
    value_net: nn.Module
    optimizer: optim.Optimizer # optimizer must be able to update policy_net, value_net, shared_net at once.
    value_loss_coef: float = 0.5
    lr_scheduler: _LRScheduler = None # Defaults to not scheduling
    grad_clip_max_norm: Union[float, None] = None
    
    def train_step(self, policy_loss: torch.Tensor, value_loss: torch.Tensor, current_epoch: int):
        loss = policy_loss + self.value_loss_coef * value_loss
        util.train_step(
            loss, 
            self.optimizer, 
            self.lr_scheduler, 
            self.grad_clip_max_norm, 
            current_epoch
        )
    
    @property
    def state_dict(self) -> dict:
        sd = {
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "value_loss_coef": self.value_loss_coef,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "grad_clip_max_norm": self.grad_clip_max_norm
        }
        return sd
        
    def load_state_dict(self, state_dict: dict):
        self.policy_net.load_state_dict(state_dict["policy_net"])
        self.value_net.load_state_dict(state_dict["value_net"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.value_loss_coef = state_dict["value_loss_coef"]
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.grad_clip_max_norm = state_dict["grad_clip_max_norm"]

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
                 entropy_coef: float = 0.0,
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
            entropy_coef (float, optional): it controls entropy regularization. Defaults to 0.0, meaning not used.
            summary_freq (int, optional): summary frequency. Defaults to 1000.
        """
        assert gamma >= 0 and gamma <= 1 and lam >= 0 and lam <= 1
        super().__init__(net_spec, policy, trajectory, clock, summary_freq)
        self.net_spec = net_spec
        self.shared_net = type(net_spec) == ActorCriticSharedNetSpec
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.num_envs = clock.num_envs
        self.value_loss_func = nn.MSELoss()
        self.device = util.get_model_device(net_spec.policy_net)
        self.action_log_probs = []
        self.entropies = []
        self.policy_losses = []
        self.value_losses = []        
        
    def select_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        pdparam = self.net_spec.policy_net(state.to(device=self.device))
        dist = self.policy.get_policy_distribution(pdparam)
        action = dist.sample()
        # save action log probability to compute policy loss
        self.action_log_probs.append(dist.log_prob(action))
        # if entropy regularization is used
        if self.entropy_coef > 0.0:
            self.entropies.append(dist.entropy())
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
        self.train_step(policy_loss, value_loss)
        # update data
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
        entropy = torch.cat(self.entropies).unsqueeze_(-1) if self.entropy_coef > 0.0 else 0.0
        policy_loss = -(advantages * log_probs + self.entropy_coef * entropy).mean()
        
        # clear already used
        self.action_log_probs.clear()
        self.entropies.clear()
        
        return policy_loss
    
    def compute_value_loss(self, v_preds: torch.Tensor, v_targets: torch.Tensor) -> torch.Tensor:
        """
        Critic
        """
        return self.value_loss_func(v_targets, v_preds)
    
    def train_step(self, policy_loss: torch.Tensor, value_loss: torch.Tensor):
        self.net_spec.train_step(policy_loss, value_loss, self.clock.training_step)
        self.clock.tick_training_step()
    
    def log_data(self, time_step: int):
        super().log_data(time_step)
        if len(self.policy_losses) > 0:
            logger.log("Network/Policy Loss", np.mean(self.policy_losses), self.clock.training_step)
            logger.log("Network/Value Loss", np.mean(self.value_losses), self.clock.training_step)
            self.policy_losses.clear()
            self.value_losses.clear()
        if self.shared_net:
            logger.log_lr_scheduler(self.net_spec.lr_scheduler, self.clock.training_step)
        else:
            logger.log_lr_scheduler(self.net_spec.policy_lr_scheduler, self.clock.training_step, "Policy Network Learning Rate")
            logger.log_lr_scheduler(self.net_spec.value_lr_scheduler, self.clock.training_step, "Value Network Learning Rate")
            