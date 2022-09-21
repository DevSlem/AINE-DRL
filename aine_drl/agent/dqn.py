from copy import copy, deepcopy
from aine_drl.agent.agent import Agent
from aine_drl.policy.policy import Policy
from aine_drl.trajectory.trajectory import Trajectory
from aine_drl.util import aine_api, logger
import aine_drl.util as util
from aine_drl.drl_util import ExperienceBatch, Clock
import aine_drl.drl_util as drl_util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from enum import Enum
from typing import Any, Union
from dataclasses import dataclass
import numpy as np

class TargetNetUpdateType(Enum):
    REPLACE = 0,
    POLYAK = 1
    
@dataclass
class DQNSpec:
    q_net: nn.Module
    target_net: nn.Module
    optimizer: optim.Optimizer
    loss_func: Any = nn.MSELoss()
    lr_scheduler: _LRScheduler = None # Defaults to not scheduling
    grad_clip_max_norm: Union[float, None] = None
    
    def train_step(self, td_loss: torch.Tensor, current_epoch: int):
        util.train_step(
            td_loss, 
            self.optimizer, 
            self.lr_scheduler, 
            self.grad_clip_max_norm, 
            current_epoch
        )
        
    @property
    def state_dict(self) -> dict:
        sd = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "grad_clip_max_norm": self.grad_clip_max_norm
        }
        return sd
    
    def load_state_dict(self, state_dict: dict):
        self.q_net.load_state_dict(state_dict["q_net"])
        self.target_net.load_state_dict(state_dict["target_net"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.grad_clip_max_norm = state_dict["grad_clip_max_norm"]


class DQN(Agent):
    """
    DQN with target network. Defaults to Vanilla DQN.
    """
    def __init__(self,
                 net_spec: DQNSpec,
                 policy: Policy,
                 trajectory: Trajectory,
                 clock: Clock,
                 gamma: float = 0.99,
                 target_net_update_type: TargetNetUpdateType = TargetNetUpdateType.REPLACE,
                 epoch: int = 3,
                 summary_freq: int = 1000,
                 **kwargs) -> None:
        """
        DQN with target network. Defaults to Vanilla DQN.

        Args:
            net_spec (DQNSpec): DQN network spec
            policy (Policy): action selection
            trajectory (Trajectory): off-policy trajectory
            clock (Clock): time step checker
            gamma (float, optional): discount factor. Defaults to 0.99.
            target_net_update_type (TargetNetUpdateType, optional): target network update type. Defaults to TargetNetUpdateType.REPLACE.
            epoch (int, optional): update count. Defaults to 3.
            **kwargs: update_freq(Defaults to 1), polyak_ratio(Defaults to 0.5)
        """
        assert gamma >= 0 and gamma <= 1
        
        self.net_spec = net_spec
        self.device = util.get_model_device(net_spec.q_net)
        self.clock = clock
        self.gamma = gamma
        self.epoch = epoch
        self.losses = []
        
        if target_net_update_type == TargetNetUpdateType.REPLACE:
            self.update_freq = kwargs["update_freq"] if "update_freq" in kwargs.keys() else 1
            assert self.update_freq >= 1
            self.net_updater = drl_util.copy_network
        elif target_net_update_type == TargetNetUpdateType.POLYAK:
            self.update_freq = 1
            polyak_ratio = kwargs["polyak_ratio"] if "polyak_ratio" in kwargs.keys() else 0.5
            assert 0 < polyak_ratio and polyak_ratio <= 1
            self.net_updater = lambda s, t: drl_util.polyak_update(s, t, polyak_ratio)
        else:
            raise ValueError
        
        super().__init__(policy, trajectory, clock, summary_freq)
        
    def select_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        pdparam = self.net_spec.q_net(state.to(device=self.device))
        dist = self.policy.get_policy_distribution(pdparam)
        return dist.sample()
    
    @aine_api
    def train(self):
        for _ in range(self.epoch):
            # update target network
            self.update_target_net()
            # compute td loss
            batch = self.trajectory.sample()
            loss = self.compute_td_loss(batch)
            # update Q network
            self.net_spec.train_step(loss, self.clock.training_step)
            self.clock.tick_training_step()
            # update data
            self.losses.append(loss.detach().cpu().numpy())
    
    @aine_api
    def log_data(self, time_step: int):
        super().log_data(time_step)
        if len(self.losses) > 0:
            logger.log("Network/TD Loss", np.mean(self.losses), self.clock.training_step)
            self.losses.clear()
        logger.log_lr_scheduler(self.net_spec.lr_scheduler, self.clock.training_step)
    
    def compute_td_loss(self, batch: ExperienceBatch) -> torch.Tensor:
        states, actions, next_states, rewards, terminateds = batch.to_tensor(self.device)
        # Q values for all actions are from the Q network
        q_values = self.net_spec.q_net(states)
        with torch.no_grad():
            # next Q values for all actions are from the target network
            next_q_target_values = self.net_spec.target_net(next_states)
        # Q value for selected action
        q_value = torch.gather(q_values, -1, actions.long().unsqueeze(-1)).squeeze(-1)
        # next maximum Q target value
        next_max_q_target_value = torch.max(next_q_target_values, -1)[0]
        # compute Q target
        q_target = rewards + self.gamma * (1 - terminateds) * next_max_q_target_value
        # compute td loss
        td_loss = self.net_spec.loss_func(q_value, q_target)
        return td_loss
    
    def update_target_net(self):
        # update target network
        if self.clock.check_time_step_freq(self.update_freq):
            self.net_updater(self.net_spec.q_net, self.net_spec.target_net)
            
    @property
    def state_dict(self) -> dict:
        sd = super().state_dict
        sd.update({"net_spec": self.net_spec.state_dict})
        return sd
    
    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.net_spec.load_state_dict(state_dict["net_spec"])

class DoubleDQN(DQN):
    """
    Double DQN.
    """
    
    def compute_td_loss(self, batch: ExperienceBatch) -> torch.Tensor:
        states, actions, next_states, rewards, terminateds = batch.to_tensor(self.device)
        # Q values for all actions are from the Q network
        q_values = self.net_spec.q_net(states)
        with torch.no_grad():
            # next Q values for all actions are from the Q network
            next_q_values = self.net_spec.q_net(next_states)
            # next Q values for all actions are from the target network
            next_q_target_values = self.net_spec.target_net(next_states)
        # Q value for selected action
        q_value = torch.gather(q_values, -1, actions.long().unsqueeze(-1)).squeeze(-1)
        # next actions with maximum Q value
        next_max_q_val_actions = torch.argmax(next_q_values, -1, keepdim=True)
        # next maximum Q target value
        next_max_q_target_value = torch.gather(next_q_target_values, -1, next_max_q_val_actions).squeeze(-1)
        # compute Q target
        q_target = rewards + self.gamma * (1 - terminateds) * next_max_q_target_value
        # compute td loss
        td_loss = self.net_spec.loss_func(q_value, q_target)
        return td_loss
