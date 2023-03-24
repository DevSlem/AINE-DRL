import copy
from dataclasses import asdict, replace

import torch
import torch.nn.functional as F

import aine_drl.drl_util as drl_util
import aine_drl.util as util
from aine_drl.agent import Agent, BehaviorType
from aine_drl.exp import Action, Experience, Observation
from aine_drl.net import NetworkTypeError, Trainer
from aine_drl.policy.policy import ActionType, Policy, PolicyActionTypeError

from .config import DoubleDQNConfig
from .net import DoubleDQNNetwork
from .trajectory import DQNExperience, DQNTrajectory


class DoubleDQN(Agent):
    """
    Double DQN with target network.
    
    Args:
        config (DoubleDQNConfig): Double DQN configuration
        network (QValueNetwork): Q value network
        policy (Policy): discrete action policy
    """
    def __init__(
        self,
        config: DoubleDQNConfig,
        network: DoubleDQNNetwork,
        trainer: Trainer,
        policy: Policy,
        num_envs: int,
        behavior_type: BehaviorType = BehaviorType.TRAIN
    ) -> None:
        if not isinstance(network, DoubleDQNNetwork):
            raise NetworkTypeError(DoubleDQNNetwork)
        if policy.action_type is not ActionType.DISCRETE:
            raise PolicyActionTypeError(ActionType.DISCRETE, policy)
        
        super().__init__(num_envs, network.device, behavior_type)
        
        self._config = config
        self._network = network
        self._target_net = copy.deepcopy(network.update_net)
        self._trainer = trainer
        self._policy = policy
        self._trajectory = DQNTrajectory(
            config.n_steps, 
            config.batch_size, 
            config.capacity, 
            num_envs,
            self.device
        )
        
        if self._config.replace_freq is not None:
            self._update_target_net = self._replace_net
        elif self._config.polyak_ratio is not None:
            self._update_target_net = self._polyak_update
        else:
            self._config = replace(self._config, replace_freq=1)
            self._update_target_net = self._replace_net
            
        self.td_loss_mean = util.IncrementalAverage()
        
    @property
    def name(self) -> str:
        return "Double DQN"
    
    def _update_train(self, exp: Experience):
        self._trajectory.add(DQNExperience(
            **asdict(exp)
        ))
        
        if self._trajectory.can_sample:
            self._train()
    
    def _update_inference(self, _: Experience):
        pass
    
    @torch.no_grad()
    def _select_action_train(self, obs: Observation) -> Action:
        # feed forward
        pdparam = self._network.forward(obs)
        
        # action sampling
        return self._policy.policy_dist(pdparam).sample()
    
    @torch.no_grad()
    def _select_action_inference(self, obs: Observation) -> Action:
        pdparam = self._network.forward(obs)
        return self._policy.policy_dist(pdparam).sample()
    
    def _train(self):
        for _ in range(self._config.epoch):
            # update target network            
            self._update_target_net()
            
            # compute td loss
            exp_batch = self._trajectory.sample()
            loss = self.compute_td_loss(exp_batch)
            
            # train step
            self._trainer.step(loss, self.training_steps)
            self._tick_training_steps()
            
            # update log data
            self.td_loss_mean.update(loss.item())
    
    def compute_td_loss(self, exp_batch: DQNExperience) -> torch.Tensor:
        # Q values for all actions are from the Q network
        q_values = self._network.forward(exp_batch.obs).discrete_pdparams
        with torch.no_grad():
            # next Q values for all actions are from the Q network
            next_q_values = self._network.forward(exp_batch.next_obs).discrete_pdparams
            # next Q values for all actions are from the target network
            next_q_target_values = self._target_net.forward(exp_batch.next_obs).discrete_pdparams
            
        actions = exp_batch.action.discrete_action.split(1, dim=1)
        
        # td loss for all action branches
        td_loss = torch.tensor(0.0, device=self.device)
        
        for i in range(exp_batch.action.num_discrete_branches):
            # Q value for the selected action
            q_value = q_values[i].gather(dim=1, index=actions[i])
            # next actions with maximum Q value
            next_max_q_action = next_q_values[i].argmax(dim=1, keepdim=True)
            # next maximum Q target value
            next_max_q_target_value = next_q_target_values[i].gather(dim=1, index=next_max_q_action)
            # compute Q target
            q_target_value = exp_batch.reward + self._config.gamma * (1 - exp_batch.terminated) * next_max_q_target_value
            # compute td loss
            td_loss += F.mse_loss(q_value, q_target_value)
            
        td_loss /= exp_batch.action.num_discrete_branches
            
        return td_loss
            
    def _replace_net(self):
        if util.check_freq(self.clock.training_step, self._config.replace_freq): # type: ignore
            drl_util.copy_module(self._network.update_net, self._target_net)
    
    def _polyak_update(self):
        drl_util.polyak_update_module(self._network.update_net, self._target_net, self._config.polyak_ratio) # type: ignore

    @property
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/TD Loss",)
    
    @property
    def log_data(self) -> dict[str, tuple]:
        ld = super().log_data
        if self.td_loss_mean.count > 0:
            ld["Network/TD Loss"] = (self.td_loss_mean.average, self.training_steps)
            self.td_loss_mean.reset()
        return ld
