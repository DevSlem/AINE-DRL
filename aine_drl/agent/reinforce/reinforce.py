import torch

import aine_drl.rl_loss as L
import aine_drl.util as util
from aine_drl.agent import Agent, BehaviorType
from aine_drl.agent.reinforce.config import REINFORCEConfig
from aine_drl.agent.reinforce.net import REINFORCENetwork
from aine_drl.agent.reinforce.trajectory import (REINFORCEExperience,
                                                 REINFORCETrajectory)
from aine_drl.exp import Action, Experience, Observation
from aine_drl.net import NetworkTypeError, Trainer


class REINFORCE(Agent):
    """
    REINFORCE with baseline. It works to only one environment. 
    
    Paper: https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf

    Args:
        config (REINFORCEConfig): REINFORCE configuration
        network (PolicyGradientNetwork): standard policy gradient network
        policy (Policy): policy
    """
    def __init__(
        self, 
        config: REINFORCEConfig,
        network: REINFORCENetwork,
        trainer: Trainer,
        behavior_type: BehaviorType = BehaviorType.TRAIN
    ) -> None:        
        if not isinstance(network, REINFORCENetwork):
            raise NetworkTypeError(REINFORCENetwork)
        
        super().__init__(1, network, config.device, behavior_type)
                
        self._config = config
        self._network = network
        self._trainer = trainer
        self._trajectory = REINFORCETrajectory()
        
        self._action_log_prob: torch.Tensor = None # type: ignore
        self._entropy: torch.Tensor = None # type: ignore
        
        self._policy_loss_mean = util.IncrementalMean()
    
    @property
    def name(self) -> str:
        return "REINFORCE"
    
    @property
    def config_dict(self) -> dict:
        return self._config.__dict__
          
    def _update_train(self, exp: Experience):
        # add the experience        
        self._trajectory.add(REINFORCEExperience(
            **exp.__dict__,
            action_log_prob=self._action_log_prob,
            entropy=self._entropy
        ))
        
        if self._trajectory.terminated:
            self._train()
    
    def _update_inference(self, _: Experience):
        pass
    
    def _select_action_train(self, obs: Observation) -> Action:
        # feed forward
        policy_dist = self._network.forward(obs)
        
        # action sampling
        action = policy_dist.sample()
        
        self._action_log_prob = policy_dist.joint_log_prob(action)
        self._entropy = policy_dist.joint_entropy()
        
        return action
    
    @torch.no_grad()
    def _select_action_inference(self, obs: Observation) -> Action:
        policy_dist = self._network.forward(obs)
        return policy_dist.sample()
            
    def _train(self):
        # batch sampling
        exp_batch = self._trajectory.sample()
        
        # compute return
        ret = L.true_return(
            exp_batch.reward.squeeze(dim=1), # (episode_len, 1) -> (episode_len,)
            self._config.gamma
        )
        
        # compute loss
        policy_loss = L.reinforce_loss(
            ret,
            exp_batch.action_log_prob.squeeze(dim=1), # (episode_len, 1) -> (episode_len,)
        )
        entropy = exp_batch.entropy.mean()
        
        # train step
        loss = policy_loss - self._config.entropy_coef * entropy
        self._trainer.step(loss, self.training_steps)
        self._tick_training_steps()
        
        # log data
        self._policy_loss_mean.update(policy_loss.item())

    @property
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/Policy Loss",)
    
    @property
    def log_data(self) -> dict[str, tuple]:
        ld = super().log_data
        if self._policy_loss_mean.count > 0:
            ld["Network/Policy Loss"] = (self._policy_loss_mean.mean, self.training_steps)
            self._policy_loss_mean.reset()
        return ld
