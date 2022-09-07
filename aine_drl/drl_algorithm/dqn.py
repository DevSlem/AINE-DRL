from aine_drl.drl_algorithm import DRLAlgorithm
from aine_drl.util import aine_api
import aine_drl.util as util
from aine_drl.drl_util import ExperienceBatch, Clock
import aine_drl.drl_util as drl_util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from enum import Enum
from typing import Any, NamedTuple
import numpy as np

class TargetNetUpdateType(Enum):
    REPLACE = 0,
    POLYAK = 1
    
class DQNSpec(NamedTuple):
    q_net: nn.Module
    target_net: nn.Module
    optimizer: optim.Optimizer
    loss_func: Any = nn.MSELoss()
    lr_scheduler: _LRScheduler = None # Defaults to not scheduling

class DQN(DRLAlgorithm):
    """
    DQN with target network. Defaults to Vanilla DQN.
    """
    def __init__(self,
                 dqn_spec: DQNSpec,
                 clock: Clock,
                 gamma: float = 0.99,
                 target_net_update_type: TargetNetUpdateType = TargetNetUpdateType.REPLACE,
                 **kwargs) -> None:
        """
        DQN with target network. Defaults to Vanilla DQN.

        Args:
            dqn_spec (DQNSpec): DQN network spec
            clock (Clock): time step checker
            gamma (float, optional): discount factor. Defaults to 0.99.
            target_net_update_type (TargetNetUpdateType, optional): target network update type. Defaults to TargetNetUpdateType.REPLACE.
            **kwargs: update_freq(Defaults to 1), polyak_ratio(Defaults to 0.5)
        """
        assert gamma >= 0 and gamma <= 1
        
        self.dqn_spec = dqn_spec
        self.device = drl_util.get_model_device(dqn_spec.q_net)
        self.clock = clock
        self.gamma = gamma
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
    
    @aine_api
    def train(self, batch: ExperienceBatch):
        self.update_target_net()
        loss = self.compute_td_loss(batch)
        # update Q network
        self.dqn_spec.optimizer.zero_grad()
        loss.backward()
        self.dqn_spec.optimizer.step()
        self.losses.append(loss.detach().cpu().numpy())
    
    @aine_api
    def get_pdparam(self, state: torch.Tensor) -> torch.Tensor:
        return self.dqn_spec.q_net(state)
    
    @aine_api
    def update_hyperparams(self, time_step: int):
        if self.dqn_spec.lr_scheduler is not None:
            self.dqn_spec.lr_scheduler.step(epoch=time_step)
    
    @aine_api
    def log_data(self, time_step: int):
        if len(self.losses) > 0:
            util.log_data("Network/TD Loss", np.mean(self.losses), time_step)
        if self.dqn_spec.lr_scheduler is not None:
            util.log_data("Network/Learning Rate", self.dqn_spec.lr_scheduler.get_lr(), time_step)
    
    def compute_td_loss(self, batch: ExperienceBatch):
        states, actions, next_states, rewards, terminateds = batch.to_tensor(self.device)
        # Q values for all actions are from the Q network
        q_values = self.dqn_spec.q_net(states)
        with torch.no_grad():
            # next Q values for all actions are from the target network
            next_q_target_values = self.dqn_spec.target_net(next_states)
        # Q value for selected action
        q_value = torch.gather(q_values, -1, actions.long().unsqueeze(-1)).squeeze(-1)
        # next maximum Q target value
        next_max_q_target_value = torch.max(next_q_target_values, -1)[0]
        # compute Q target
        q_target = rewards + self.gamma * (1 - terminateds) * next_max_q_target_value
        # compute td loss
        td_loss = self.dqn_spec.loss_func(q_value, q_target)
        return td_loss
    
    def update_target_net(self):
        # update target network
        if self.clock.check_time_step_freq(self.update_freq):
            self.net_updater(self.dqn_spec.q_net, self.dqn_spec.target_net)

class DoubleDQN(DQN):
    """
    Double DQN.
    """
    def __init__(self, 
                 dqn_spec: DQNSpec, 
                 clock: Clock, 
                 gamma: float = 0.99, 
                 target_net_update_type: TargetNetUpdateType = TargetNetUpdateType.REPLACE, 
                 **kwargs) -> None:
        """
        Double DQN.

        Args:
            dqn_spec (DQNSpec): DQN network spec
            clock (Clock): time step checker
            gamma (float, optional): discount factor. Defaults to 0.99.
            target_net_update_type (TargetNetUpdateType, optional): target network update type. Defaults to TargetNetUpdateType.REPLACE.
            **kwargs: update_freq(Defaults to 1), polyak_ratio(Defaults to 0.5)
        """
        super().__init__(dqn_spec, clock, gamma, target_net_update_type, **kwargs)
    
    def compute_td_loss(self, batch: ExperienceBatch):
        states, actions, next_states, rewards, terminateds = batch.to_tensor(self.device)
        # Q values for all actions are from the Q network
        q_values = self.dqn_spec.q_net(states)
        with torch.no_grad():
            # next Q values for all actions are from the Q network
            next_q_values = self.dqn_spec.q_net(next_states)
            # next Q values for all actions are from the target network
            next_q_target_values = self.dqn_spec.target_net(next_states)
        # Q value for selected action
        q_value = torch.gather(q_values, -1, actions.long().unsqueeze(-1)).squeeze(-1)
        # next actions with maximum Q value
        next_max_q_val_actions = torch.argmax(next_q_values, -1, keepdim=True)
        # next maximum Q target value
        next_max_q_target_value = torch.gather(next_q_target_values, -1, next_max_q_val_actions).squeeze(-1)
        # compute Q target
        q_target = rewards + self.gamma * (1 - terminateds) * next_max_q_target_value
        # compute td loss
        td_loss = self.dqn_spec.loss_func(q_value, q_target)
        return td_loss
