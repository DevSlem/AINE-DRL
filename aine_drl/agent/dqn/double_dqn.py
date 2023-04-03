import copy
from dataclasses import replace

import torch
import torch.nn.functional as F

import aine_drl.util as util
import aine_drl.util.func as util_f
from aine_drl.agent import Agent, BehaviorType
from aine_drl.agent.dqn.config import DoubleDQNConfig
from aine_drl.agent.dqn.net import DoubleDQNNetwork
from aine_drl.agent.dqn.trajectory import DQNExperience, DQNTrajectory
from aine_drl.exp import Action, Experience, Observation
from aine_drl.net import NetworkTypeError, Trainer


class DoubleDQN(Agent):
    """
    Double DQN with target network. 
    
    Paper: https://arxiv.org/abs/1509.06461
    """
    def __init__(
        self,
        config: DoubleDQNConfig,
        network: DoubleDQNNetwork,
        trainer: Trainer,
        num_envs: int,
        behavior_type: BehaviorType = BehaviorType.TRAIN
    ) -> None:
        if not isinstance(network, DoubleDQNNetwork):
            raise NetworkTypeError(DoubleDQNNetwork)
        
        super().__init__(num_envs, network, config.device, behavior_type)
        
        self._config = config
        self._network = network
        self._target_net = copy.deepcopy(network)
        self._trainer = trainer
        
        replay_buffer_device = self.device if self._config.replay_buffer_device is None else torch.device(self._config.replay_buffer_device)
        
        self._trajectory = DQNTrajectory(
            config.n_steps, 
            config.batch_size, 
            config.capacity, 
            num_envs,
            replay_buffer_device
        )
        
        if self._config.replace_freq is not None:
            self._update_target_net = self._replace_net
        elif self._config.polyak_ratio is not None:
            self._update_target_net = self._polyak_update
        else:
            self._config = replace(self._config, replace_freq=1)
            self._update_target_net = self._replace_net
            
        self.td_loss_mean = util.IncrementalMean()
        
    @property
    def name(self) -> str:
        return "Double DQN"
    
    @property
    def config_dict(self) -> dict:
        return self._config.__dict__
    
    def _update_train(self, exp: Experience):
        self._trajectory.add(DQNExperience(
            **exp.__dict__
        ))
        
        if self._trajectory.can_sample:
            self._train()
    
    def _update_inference(self, _: Experience):
        pass
    
    @torch.no_grad()
    def _select_action_train(self, obs: Observation) -> Action:
        # feed forward
        policy_dist, _ = self._network.forward(obs)
        
        # action sampling
        return policy_dist.sample()
    
    @torch.no_grad()
    def _select_action_inference(self, obs: Observation) -> Action:
        policy_dist, _ = self._network.forward(obs)
        return policy_dist.sample()
    
    def _train(self):
        for _ in range(self._config.epoch):
            # update target network            
            self._update_target_net()
            
            # compute td loss
            exp_batch = self._trajectory.sample(self.device)
            loss = self._compute_td_loss(exp_batch)
            
            # train step
            self._trainer.step(loss, self.training_steps)
            self._tick_training_steps()
            
            # update log data
            self.td_loss_mean.update(loss.item())
    
    def _compute_td_loss(self, exp_batch: DQNExperience) -> torch.Tensor:
        # Q values for all actions are from the Q network
        _, q_values = self._network.forward(exp_batch.obs)
        with torch.no_grad():
            # next Q values for all actions are from the Q network
            _, next_q_values = self._network.forward(exp_batch.next_obs)
            # next Q values for all actions are from the target network
            _, next_q_target_values = self._target_net.forward(exp_batch.next_obs)
            
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
        if self.training_steps % self._config.replace_freq == 0: # type: ignore
            util_f.copy_module(self._network.model(), self._target_net.model())
    
    def _polyak_update(self):
        util_f.polyak_update_module(self._network.model(), self._target_net.model(), self._config.polyak_ratio) # type: ignore

    @property
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/TD Loss",)
    
    @property
    def log_data(self) -> dict[str, tuple]:
        ld = super().log_data
        if self.td_loss_mean.count > 0:
            ld["Network/TD Loss"] = (self.td_loss_mean.mean, self.training_steps)
            self.td_loss_mean.reset()
        return ld
