from aine_drl.agent.agent import Agent
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from aine_drl.drl_util.clock import Clock
from aine_drl.drl_util.experience import ExperienceBatch
from aine_drl.policy.policy import Policy
from aine_drl.trajectory.montecarlo_trajectory import MonteCarloTrajectory
import aine_drl.util as util
import aine_drl.drl_util as drl_util
from aine_drl.util.decorator import aine_api
import numpy as np

class REINFORCENetSpec(NamedTuple):
    policy_net: nn.Module
    optimizer: optim.Optimizer
    lr_scheduler: _LRScheduler = None # Defaults to not scheduling

class REINFORCE(Agent):
    """
    REINFORCE with Baseline. It's a Monte Carlo method. It allows multiple environments but not recommended beacause of the stability. 
    """
    def __init__(self, 
                 net_spec: REINFORCENetSpec,
                 policy: Policy, 
                 trajectory: MonteCarloTrajectory, 
                 clock: Clock, 
                 gamma: float = 0.99,
                 summary_freq: int = 1000) -> None:
        """
        REINFORCE with Baseline. It's a Monte Carlo method. It allows multiple environments but not recommended beacause of the stability. 

        Args:
            net_spec (REINFORCENetSpec): REINFORCE network spec
            policy (Policy): policy of either discrete or continuous action
            trajectory (MonteCarloTrajectory): trajectory for the Monte Carlo method 
            clock (Clock): time step checker
            gamma (float, optional): discount factor. Defaults to 0.99.
            summary_freq (int, optional): summary frequency. Defaults to 1000.
        """
        assert gamma >= 0 and gamma <= 1 and isinstance(trajectory, MonteCarloTrajectory)
        super().__init__(policy, trajectory, clock, summary_freq)
        self.net_spec = net_spec
        self.gamma = gamma
        self.device = util.get_model_device(net_spec.policy_net)
        self.action_log_probs = []
        self.losses = []
    
    def select_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        pdparam = self.net_spec.policy_net(state.to(device=self.device))
        dist = self.policy.get_policy_distribution(pdparam)
        action = dist.sample()
        # save action log probability to compute policy loss
        self.action_log_probs.append(dist.log_prob(action))
        return action
    
    @aine_api
    def train(self):
        # compute policy loss
        batch = self.trajectory.sample()
        loss = self.compute_policy_loss(batch)
        # gradient descent
        self.net_spec.optimizer.zero_grad()
        loss.backward()
        self.net_spec.optimizer.step()
        # update data
        self.losses.append(loss.detach().cpu().item())
        self.clock.tick_training_step()
        util.lr_scheduler_step(self.net_spec.lr_scheduler, self.clock.training_step)
        self.action_log_probs.clear()
    
    def compute_policy_loss(self, batch: ExperienceBatch):
        eps = torch.finfo(torch.float32).eps
        states, actions, next_states, rewards, terminateds = batch.to_tensor(self.device)
        # calculate returns at all time steps
        returns = drl_util.calc_returns(rewards, terminateds, self.gamma)
        # normalize returns (REINFORCE with Baseline)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        log_probs = self.flattened_action_log_probs(terminateds)
        # compute policy loss
        policy_loss = -(returns * log_probs).sum()
        return policy_loss
            
    def flattened_action_log_probs(self, terminateds: torch.Tensor) -> torch.Tensor:
        terminated_idx = terminateds.nonzero()
        epi_lengths = [terminated_idx[0] + 1]
        for i in range(len(terminated_idx) - 1):
            epi_lengths.append(terminated_idx[i + 1] - terminated_idx[i])
        action_log_probs = torch.stack(self.action_log_probs)
        if len(epi_lengths) > 1:
            action_log_probs = action_log_probs.T
        else:
            action_log_probs.unsqueeze_(0)
        flattend_action_log_probs = []
        for i, epi_len in enumerate(epi_lengths):
            flattend_action_log_probs.append(action_log_probs[i][:epi_len])
        return torch.cat(flattend_action_log_probs)
    
    def log_data(self, time_step: int):
        super().log_data(time_step)
        if len(self.losses) > 0:
            util.log_data("Network/Policy Loss", np.mean(self.losses), time_step)
        if self.net_spec.lr_scheduler is not None:
            lr = self.net_spec.lr_scheduler.get_lr()
            util.log_data("Network/Learning Rate", lr if type(lr) is float else lr[0], time_step)