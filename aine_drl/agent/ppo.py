from typing import Union
from aine_drl.agent.a2c import A2C, ActorCriticNetSpec, ActorCriticSharedNetSpec
from aine_drl.drl_util.clock import Clock
from aine_drl.policy.policy import Policy
from aine_drl.trajectory.trajectory import Trajectory
import torch

class PPO(A2C):
    """
    Proximal Policy Optimization (PPO). See details in https://arxiv.org/abs/1707.06347.
    """
    def __init__(self, 
                 net_spec: Union[ActorCriticNetSpec, ActorCriticSharedNetSpec], 
                 policy: Policy, 
                 trajectory: Trajectory, 
                 clock: Clock, 
                 gamma: float = 0.99, 
                 lam: float = 0.95, 
                 epsilon_clip: float = 0.2,
                 entropy_coef: float = 0,
                 epoch: int = 3,
                 summary_freq: int = 1000) -> None:
        """
        Proximal Policy Optimization (PPO). See details in https://arxiv.org/abs/1707.06347.

        Args:
            net_spec (ActorCriticNetSpec | ActorCriticSharedNetSpec): network spec
            policy (Policy): policy for action selection
            trajectory (Trajectory): on-policy trajectory for sampling
            clock (Clock): time step checker
            gamma (float, optional): discount factor. Defaults to 0.99.
            lam (float, optional): lambda which controls the GAE balanace between bias and variance. Defaults to 0.95.
            epsilon_clip (float, optional): clipping the probability ratio (pi_theta / pi_theta_old) to [1-eps, 1+eps]. Defaults to 0.2.
            entropy_coef (float, optional): it controls entropy regularization. Defaults to 0.0, meaning not used.
            epoch (int, optional): network update count when train() method is called. Defaults to 3.
            summary_freq (int, optional): summary frequency. Defaults to 1000.
        """
        super().__init__(net_spec, policy, trajectory, clock, gamma, lam, entropy_coef, summary_freq)
        self.epsilon_clip = epsilon_clip
        self.epoch = epoch
        
    def select_action_tensor(self, state: torch.Tensor) -> torch.Tensor:
        pdparam = self.net_spec.policy_net(state.to(device=self.device))
        dist = self.policy.get_policy_distribution(pdparam)
        action = dist.sample()
        # save action log probability to compute policy loss
        self.action_log_probs.append(dist.log_prob(action))
        return action
        
    def train(self):
        batch = self.trajectory.sample()
        states, actions, next_states, rewards, terminateds = batch.to_tensor(self.device)
        # block to flow gradients of old policy
        old_log_probs = torch.cat(self.action_log_probs).unsqueeze_(-1).detach_()
        # compute advantages and v_targets
        with torch.no_grad():
            v_preds = self.net_spec.value_net(states)
        advantages, v_targets = self.compute_advantage_v_target(v_preds, next_states, rewards, terminateds)
        # update network
        for _ in range(self.epoch):
            v_preds, log_probs, entropies = self.evaluate(states, actions)
            policy_loss = self.compute_policy_loss(advantages, old_log_probs, log_probs, entropies)
            value_loss = self.compute_value_loss(v_preds, v_targets)
            self.train_step(policy_loss, value_loss)
        # clear saved old action log probilities
        self.action_log_probs.clear()
        # update data
        self.policy_losses.append(policy_loss.detach().cpu().item())
        self.value_losses.append(value_loss.detach().cpu().item())
        
    def compute_policy_loss(self, 
                            advantages: torch.Tensor, 
                            old_log_probs: torch.Tensor,
                            log_probs: torch.Tensor,
                            entropies: Union[torch.Tensor, float] = 0.0):
        assert not old_log_probs.requires_grad, "gradients of log_probs only flows."
        # pi_theta / pi_theta_old
        ratios = torch.exp(log_probs - old_log_probs)
        # surrogate loss
        advantages = torch.broadcast_to(advantages.squeeze(-1), ratios.shape[::-1]).T
        sur1 = ratios * advantages
        sur2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
        # compute loss
        loss = -(torch.min(sur1, sur2) + self.entropy_coef * entropies).mean()
        return loss
    
    def evaluate(self,
                 states: torch.Tensor,
                 actions: torch.Tensor):
        v_preds = self.net_spec.value_net(states)
        pdparam = self.net_spec.policy_net(states)
        dist = self.policy.get_policy_distribution(pdparam)
        log_probs = dist.log_prob(actions).unsqueeze_(-1)
        entropies = dist.entropy().unsqueeze_(-1)
        return v_preds, log_probs, entropies
