from aine_drl.agent import Agent
from aine_drl.experience import ActionTensor, Experience
from aine_drl.network import NetworkTypeError
from aine_drl.policy.policy import Policy
from .config import A2CConfig
from .net import A2CSharedNetwork
from .a2c_trajectory import A2CExperienceBatch, A2CTrajectory
import aine_drl.drl_util as drl_util
import aine_drl.util as util
import torch
import torch.nn.functional as F

class A2C(Agent):
    """
    Advantage Actor Critic (A2C).

    Args:
        config (A2CConfig): A2C configuration
        network (ActorCriticSharedNetwork): standard actor critic network
        policy (Policy): policy
        num_envs (int): number of environments
    """
    def __init__(self, 
                 config: A2CConfig,
                 network: A2CSharedNetwork,
                 policy: Policy,
                 num_envs: int) -> None:        
        if not isinstance(network, A2CSharedNetwork):
            raise NetworkTypeError(A2CSharedNetwork)
        
        super().__init__(network, policy, num_envs)
        
        self.config = config
        self.network = network
        self.trajectory = A2CTrajectory(self.config.training_freq)
        
        self.current_action_log_prob = None
        self.v_pred = None
        self.entropy = None
        
        self.actor_average_loss = util.IncrementalAverage()
        self.critic_average_loss = util.IncrementalAverage()    
        
    @property
    def name(self) -> str:
        return "A2C"
        
    def update(self, experience: Experience):
        super().update(experience)
        
        # add the experience
        self.trajectory.add(
            experience,
            self.current_action_log_prob,
            self.v_pred,
            self.entropy
        )
        
        # if training frequency is reached, start training
        if self.trajectory.count == self.config.training_freq:
            self.train()
            
    def select_action_train(self, obs: torch.Tensor) -> ActionTensor:
        # feed forward 
        pdparam, v_pred = self.network.forward(obs)
        
        # action sampling
        dist = self.policy.policy_dist(pdparam)
        action = dist.sample()
        
        # store data
        self.current_action_log_prob = dist.joint_log_prob(action).cpu()
        self.v_pred = v_pred.cpu()
        self.entropy = dist.joint_entropy().cpu()
        
        return action
    
    def select_action_inference(self, obs: torch.Tensor) -> ActionTensor:
        pdparam, _ = self.network.forward(obs)
        dist = self.policy.policy_dist(pdparam)
        return dist.sample()
            
    def train(self):
        # batch sampling
        exp_batch = self.trajectory.sample(self.device)
        
        # compute actor critic loss
        advantage, v_target = self.compute_adavantage_v_target(exp_batch)
        actor_loss = A2C.compute_actor_loss(advantage, exp_batch.action_log_prob)
        critic_loss = A2C.compute_critic_loss(exp_batch.v_pred, v_target)
        entropy = exp_batch.entropy.mean()
        
        # train step
        loss = actor_loss + self.config.value_loss_coef * critic_loss - self.config.entropy_coef * entropy
        self.network.train_step(loss, self.clock.training_step)
        self.clock.tick_training_step()
        
        # log data
        self.actor_average_loss.update(actor_loss.item())
        self.critic_average_loss.update(critic_loss.item())

        
    def compute_adavantage_v_target(self, exp_batch: A2CExperienceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantage, v_target. `batch_size` is `num_evns` x `n_steps`.

        Args:
            exp_batch (A2CExperienceBatch): experience batch

        Returns:
            tuple[Tensor, Tensor]: advantage, v_target whose each shape is `(batch_size, 1)`
        """
        with torch.no_grad():
            final_next_obs = exp_batch.next_obs[-self.num_envs:]
            _, final_next_v_pred = self.network.forward(final_next_obs)
        
        v_pred = torch.cat([exp_batch.v_pred.detach(), final_next_v_pred])
        
        # (num_envs * n + 1, 1) -> (num_envs, n, 1) -> (num_envs, n)
        v_pred = drl_util.batch2perenv(v_pred, self.num_envs).squeeze_(-1)
        reward = drl_util.batch2perenv(exp_batch.reward, self.num_envs).squeeze_(-1)
        terminated = drl_util.batch2perenv(exp_batch.terminated, self.num_envs).squeeze_(-1)
        
        # compute advantage using GAE
        advantage = drl_util.compute_gae(
            v_pred,
            reward,
            terminated,
            self.config.gamma,
            self.config.lam
        )
        
        # compute v_target
        v_target = advantage + v_pred[:, :-1]
        
        advantage = drl_util.perenv2batch(advantage.unsqueeze_(-1))
        v_target = drl_util.perenv2batch(v_target.unsqueeze_(-1))
        
        return advantage, v_target
    
    @staticmethod
    def compute_actor_loss(advantage: torch.Tensor, 
                           action_log_prob: torch.Tensor) -> torch.Tensor:
        """
        Compute actor loss using A2C. It uses mean loss not sum loss.

        Args:
            advantage (Tensor): whose shape is `(batch_size, 1)`
            action_log_prob (Tensor): log(pi) whose shape is `(batch_size, 1)`

        Returns:
            Tensor: A2C actor loss
        """        
        # compute actor loss
        loss = -(advantage * action_log_prob).mean()
        return loss
    
    @staticmethod
    def compute_critic_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        """
        Compute critic loss using MSE.

        Args:
            v_pred (Tensor): predicted state value whose gradient flows and shape is `(batch_size, 1)`
            v_target (Tensor): target state value whose gradient never flows and shape is `(batch_size, 1)`

        Returns:
            Tensor: critic loss
        """
        return F.mse_loss(v_target, v_pred)

    @property
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/Actor Loss", "Network/Critic Loss")
    
    @property
    def log_data(self) -> dict[str, tuple]:
        ld = super().log_data
        if self.actor_average_loss.count > 0:
            ld["Network/Actor Loss"] = (self.actor_average_loss.average, self.clock.training_step)
            ld["Network/Critic Loss"] = (self.critic_average_loss.average, self.clock.training_step)
            self.actor_average_loss.reset()
            self.critic_average_loss.reset()
        return ld
