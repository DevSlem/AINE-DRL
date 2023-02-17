from aine_drl.agent import Agent
from aine_drl.experience import ActionTensor, Experience
from aine_drl.policy.policy import Policy
from aine_drl.network import NetworkTypeError
from .config import REINFORCEConfig
from .net import REINFORCENetwork
from .reinforce_trajectory import REINFORCEExperienceBatch, REINFORCETrajectory
import aine_drl.drl_util as drl_util
import aine_drl.util as util
import torch

class REINFORCE(Agent):
    """
    REINFORCE with baseline. It works to only one environment. See details in https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf.

    Args:
        config (REINFORCEConfig): REINFORCE configuration
        network (PolicyGradientNetwork): standard policy gradient network
        policy (Policy): policy
    """
    def __init__(self, 
                 config: REINFORCEConfig,
                 network: REINFORCENetwork,
                 policy: Policy) -> None:        
        if not isinstance(network, REINFORCENetwork):
            raise NetworkTypeError(REINFORCENetwork)
        
        super().__init__(network, policy, num_envs=1)
        
        self.config = config
        self.network = network
        self.trajectory = REINFORCETrajectory()
        
        self.current_action_log_prob = None
        self.entropy = None
        
        self.policy_average_loss = util.IncrementalAverage()
    
    def update(self, experience: Experience):
        super().update(experience)
        
        # add the experience
        self.trajectory.add(
            experience,
            self.current_action_log_prob,
            self.entropy
        )
        
        # if the environment is terminated
        if experience.terminated.item() > 0.5:
            self.train()
            
    def select_action_train(self, obs: torch.Tensor) -> ActionTensor:
        # feed forward 
        pdparam = self.network.forward(obs)
        
        # action sampling
        dist = self.policy.get_policy_distribution(pdparam)
        action = dist.sample()
        
        # store data
        self.current_action_log_prob = dist.joint_log_prob(action).cpu()
        self.entropy = dist.joint_entropy().cpu()
        
        return action
    
    def select_action_inference(self, obs: torch.Tensor) -> ActionTensor:
        pdparam = self.network.forward(obs)
        dist = self.policy.get_policy_distribution(pdparam)
        return dist.sample()
            
    def train(self):
        # batch sampling
        exp_batch = self.trajectory.sample(self.device)
        
        # compute policy loss
        returns = self.compute_return(exp_batch)
        policy_loss = REINFORCE.compute_policy_loss(returns, exp_batch.action_log_prob)
        entropy = exp_batch.entropy.mean()
        
        # train step
        loss = policy_loss - self.config.entropy_coef * entropy
        self.network.train_step(loss, self.clock.training_step)
        self.clock.tick_training_step()
        
        # log data
        self.policy_average_loss.update(policy_loss.item())

        
    def compute_return(self, exp_batch: REINFORCEExperienceBatch) -> torch.Tensor:
        """
        Compute return.

        Args:
            exp_batch (REINFORCEExperienceBatch): experience batch

        Returns:
            Tensor: return whose shape is `(episode_len, 1)`
        """        
        # compute advantage using GAE
        returns = drl_util.compute_return(
            exp_batch.reward.squeeze(dim=1), # (episode_len, 1) -> (episode_len,)
            self.config.gamma
        )
          
        # (episode_len,) -> (episode_len, 1)
        return returns.unsqueeze_(dim=1)
    
    @staticmethod
    def compute_policy_loss(returns: torch.Tensor, 
                           action_log_prob: torch.Tensor) -> torch.Tensor:
        """
        Compute policy loss using REINFORCE. It uses mean loss not sum loss.

        Args:
            returns (Tensor): whose shape is `(batch_size, 1)`
            action_log_prob (Tensor): log(pi) whose shape is `(batch_size, 1)`

        Returns:
            Tensor: REINFORCE policy loss
        """        
        # compute policy loss
        eps = torch.finfo(torch.float32).eps
        returns = (returns - returns.mean()) / (returns.std() + eps)
        loss = -(returns * action_log_prob).mean()
        return loss

    @property
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/Policy Loss",)
    
    @property
    def log_data(self) -> dict[str, tuple]:
        ld = super().log_data
        if self.policy_average_loss.count > 0:
            ld["Network/Policy Loss"] = (self.policy_average_loss.average, self.clock.training_step)
            self.policy_average_loss.reset()
        return ld
