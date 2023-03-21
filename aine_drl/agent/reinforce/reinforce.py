from dataclasses import asdict

import torch

import aine_drl.rl_loss as L
import aine_drl.util as util
from aine_drl.agent import Agent, BehaviorType
from aine_drl.exp import Action, Experience
from aine_drl.net import NetworkTypeError
from aine_drl.policy.policy import Policy

from .config import REINFORCEConfig
from .net import REINFORCENetwork, REINFORCEOptim
from .trajectory import REINFORCEExperience, REINFORCETrajectory


class REINFORCE(Agent):
    """
    REINFORCE with baseline. It works to only one environment. 
    
    See details in https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf.

    Args:
        config (REINFORCEConfig): REINFORCE configuration
        network (PolicyGradientNetwork): standard policy gradient network
        policy (Policy): policy
    """
    def __init__(
        self, 
        config: REINFORCEConfig,
        network: REINFORCENetwork,
        optimizer: REINFORCEOptim,
        policy: Policy,
        behavior_type: BehaviorType = BehaviorType.TRAIN
    ) -> None:        
        if not isinstance(network, REINFORCENetwork):
            raise NetworkTypeError(REINFORCENetwork)
        
        super().__init__(network, policy, 1, behavior_type)
                
        self._config = config
        self._network = network
        self._optimizer = optimizer
        self._trajectory = REINFORCETrajectory()
        
        self._action_log_prob: torch.Tensor = None # type: ignore
        self._entropy: torch.Tensor = None # type: ignore
        
        self._policy_loss_mean = util.IncrementalAverage()
    
    @property
    def name(self) -> str:
        return "REINFORCE"
          
    def _update_train(self, exp: Experience):
        # add the experience        
        self._trajectory.add(REINFORCEExperience(
            **asdict(exp),
            action_log_prob=self._action_log_prob,
            entropy=self._entropy
        ))
        
        if self._trajectory.terminated:
            self._train()
    
    def _update_inference(self, _: Experience):
        pass
    
    def _select_action_train(self, obs: torch.Tensor) -> Action:
        # feed forward
        pdparam = self._network.forward(obs)
        
        # action sampling
        dist = self._policy.policy_dist(pdparam)
        action = dist.sample()
        
        self._action_log_prob = dist.joint_log_prob(action)
        self._entropy = dist.joint_entropy()
        
        return action
    
    @torch.no_grad()
    def _select_action_inference(self, obs: torch.Tensor) -> Action:
        pdparam = self._network.forward(obs)
        return self._policy.policy_dist(pdparam).sample()
            
    def _train(self):
        # batch sampling
        exp_batch = self._trajectory.sample()
        
        # compute return
        ret = L.true_return(
            exp_batch.reward.squeeze(dim=1), # (episode_len, 1) -> (episode_len,)
            self._config.gamma
        ).unsqueeze_(dim=1)
        
        # compute loss
        policy_loss = L.reinforce_loss(
            ret,
            exp_batch.action_log_prob.squeeze(dim=1), # (episode_len, 1) -> (episode_len,)
        )
        entropy = exp_batch.entropy.mean()
        
        # train step
        loss = policy_loss - self._config.entropy_coef * entropy
        self._optimizer.step(loss, self.clock.training_step)
        self.clock.tick_training_step()
        
        # log data
        self._policy_loss_mean.update(policy_loss.item())

    @property
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/Policy Loss",)
    
    @property
    def log_data(self) -> dict[str, tuple]:
        ld = super().log_data
        if self._policy_loss_mean.count > 0:
            ld["Network/Policy Loss"] = (self._policy_loss_mean.average, self.clock.training_step)
            self._policy_loss_mean.reset()
        return ld
