from typing import Dict
from aine_drl.agent import Agent
from aine_drl.experience import ActionTensor, Experience, ExperienceBatchTensor
from aine_drl.policy.policy import ActionType, Policy
from aine_drl.network import NetworkTypeError
from aine_drl.trajectory.experience_replay import ExperienceReplay
from .config import DoubleDQNConfig
from .net import DoubleDQNNetwork
import aine_drl.drl_util as drl_util
import aine_drl.util as util
import torch
import torch.nn.functional as F
import copy

class DoubleDQN(Agent):
    """
    Double DQN with target network.
    
    Args:
        config (DoubleDQNConfig): Double DQN configuration
        network (QValueNetwork): Q value network
        policy (Policy): discrete action policy
    """
    def __init__(self,
                 config: DoubleDQNConfig,
                 network: DoubleDQNNetwork,
                 policy: Policy,
                 num_envs: int) -> None:
        if not isinstance(network, DoubleDQNNetwork):
            raise NetworkTypeError(DoubleDQNNetwork)
        if policy.action_type is not ActionType.DISCRETE:
            raise TypeError(f"The policy must be discrete action policy, but \"{type(policy).__name__}\" is \"{policy.action_type}\".")
        
        super().__init__(network, policy, num_envs)
        
        self.config = config
        self.network = network
        self.target_network = copy.deepcopy(network)
        self.trajectory = ExperienceReplay(
            config.training_freq, 
            config.batch_size, 
            config.capacity, 
            num_envs,
            online=True
        )
        
        if self.config.replace_freq is not None:
            self.update_target_network = self._replace_net
        elif self.config.polyak_ratio is not None:
            self.update_target_network = self._polyak_update
        else:
            self.config = config._replace(replace_freq=1)
            self.update_target_network = self._replace_net
            
        self.average_td_loss = util.IncrementalAverage()
        
    def select_action_train(self, obs: torch.Tensor) -> ActionTensor:
        # feed forward
        pdparam = self.network.forward(obs)
        
        # action sampling
        dist = self.policy.get_policy_distribution(pdparam)
        action = dist.sample()
        
        return action
    
    def select_action_inference(self, obs: torch.Tensor) -> ActionTensor:
        pdparam = self.network.forward(obs)
        dist = self.policy.get_policy_distribution(pdparam)
        return dist.sample()
    
    def update(self, experience: Experience):
        super().update(experience)
        
        self.trajectory.add(experience)
        
        if self.trajectory.can_sample:
            self.train()
    
    def train(self):
        for _ in range(self.config.epoch):
            # update target network            
            self.update_target_network()
            
            # compute td loss
            exp_batch = self.trajectory.sample(self.device)
            td_loss = self.compute_td_loss(exp_batch)
            
            # train step
            self.network.train_step(td_loss, self.clock.training_step)
            self.clock.tick_training_step()
            
            # update log data
            self.average_td_loss.update(td_loss.item())
    
    def compute_td_loss(self, exp_batch: ExperienceBatchTensor) -> torch.Tensor:
        # Q values for all actions are from the Q network
        q_values = self.network.forward(exp_batch.obs).discrete_pdparams
        with torch.no_grad():
            # next Q values for all actions are from the Q network
            next_q_values = self.network.forward(exp_batch.next_obs).discrete_pdparams
            # next Q values for all actions are from the target network
            next_q_target_values = self.target_network.forward(exp_batch.next_obs).discrete_pdparams
            
        actions = exp_batch.action.discrete_action.split(1, dim=1)
        
        # td loss for all action branches
        td_loss = 0.0
        
        for i in range(exp_batch.action.num_discrete_branches):
            # Q value for the selected action
            q_value = q_values[i].gather(dim=1, index=actions[i])
            # next actions with maximum Q value
            next_max_q_action = next_q_values[i].argmax(dim=1, keepdim=True)
            # next maximum Q target value
            next_max_q_target_value = next_q_target_values[i].gather(dim=1, index=next_max_q_action)
            # compute Q target
            q_target_value = exp_batch.reward + self.config.gamma * (1 - exp_batch.terminated) * next_max_q_target_value
            # compute td loss
            td_loss += F.mse_loss(q_value, q_target_value)
            
        td_loss /= exp_batch.action.num_discrete_branches
            
        return td_loss
            
    def _replace_net(self):
        if util.check_freq(self.clock.training_step, self.config.replace_freq): # type: ignore
            drl_util.copy_network(self.network, self.target_network)
    
    def _polyak_update(self):
        drl_util.polyak_update_network(self.network, self.target_network, self.config.polyak_ratio) # type: ignore

    @property
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/TD Loss",)
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        ld = super().log_data
        if self.average_td_loss.count > 0:
            ld["Network/TD Loss"] = (self.average_td_loss.average, self.clock.training_step)
            self.average_td_loss.reset()
        return ld
