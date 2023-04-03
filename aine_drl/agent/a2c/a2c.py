import torch

import aine_drl.rl_loss as L
import aine_drl.util as util
from aine_drl.agent import Agent, BehaviorType
from aine_drl.agent.a2c.config import A2CConfig
from aine_drl.agent.a2c.net import A2CSharedNetwork
from aine_drl.agent.a2c.trajectory import A2CExperience, A2CTrajectory
from aine_drl.exp import Action, Experience, Observation
from aine_drl.net import NetworkTypeError, Trainer
from aine_drl.util.func import batch2perenv, perenv2batch


class A2C(Agent):
    """
    Advantage Actor Critic (A2C).
    """
    def __init__(
        self, 
        config: A2CConfig,
        network: A2CSharedNetwork,
        trainer: Trainer,
        num_envs: int,
        behavior_type: BehaviorType = BehaviorType.TRAIN
    ) -> None:        
        if not isinstance(network, A2CSharedNetwork):
            raise NetworkTypeError(A2CSharedNetwork)
        
        super().__init__(num_envs, network, config.device, behavior_type)
        
        self._config = config
        self._network = network
        self._trainer = trainer
        self._trajectory = A2CTrajectory(self._config.n_steps)
        
        self._action_log_prob: torch.Tensor = None # type: ignore
        self._state_value: torch.Tensor = None # type: ignore
        self._entropy: torch.Tensor = None # type: ignore
        
        self.actor_average_loss = util.IncrementalMean()
        self.critic_average_loss = util.IncrementalMean()    
        
    @property
    def name(self) -> str:
        return "A2C"
    
    @property
    def config_dict(self) -> dict:
        return self._config.__dict__
                
    def _update_train(self, exp: Experience):
        self._trajectory.add(A2CExperience(
            **exp.__dict__,
            action_log_prob=self._action_log_prob,
            state_value=self._state_value,
            entropy=self._entropy
        ))
        
        if self._trajectory.reached_n_steps:
            self._train()
    
    def _update_inference(self, _: Experience):
        pass
    
    def _select_action_train(self, obs: Observation) -> Action:
        # feed forward
        policy_dist, state_value = self._network.forward(obs)
        
        # action sampling
        action = policy_dist.sample()
        
        self._action_log_prob = policy_dist.joint_log_prob(action)
        self._state_value = state_value
        self._entropy = policy_dist.joint_entropy()
        
        return action
    
    @torch.no_grad()
    def _select_action_inference(self, obs: Observation) -> Action:
        policy_dist, _ = self._network.forward(obs)
        return policy_dist.sample()
            
    def _train(self):
        # batch sampling
        exp_batch = self._trajectory.sample()
        
        # compute advantage and target state value
        advantage, target_state_value = self._compute_adv_target(exp_batch)
        
        # compute actor critic loss
        actor_loss = L.advantage_policy_loss(advantage, exp_batch.action_log_prob)
        critic_loss = L.bellman_value_loss(exp_batch.state_value, target_state_value)
        entropy = exp_batch.entropy.mean()
        
        # train step
        loss = actor_loss + self._config.value_loss_coef * critic_loss - self._config.entropy_coef * entropy
        self._trainer.step(loss, self.training_steps)
        self._tick_training_steps()
        
        # log data
        self.actor_average_loss.update(actor_loss.item())
        self.critic_average_loss.update(critic_loss.item())
        
    def _compute_adv_target(self, exp_batch: A2CExperience) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantage and target state value. 
        
        `batch_size` is `num_evns` * `n_steps`.

        Args:
            exp_batch (A2CExperience): experience batch

        Returns:
            advantage (Tensor): `(batch_size, 1)`
            target_state_value (Tensor): `(batch_size, 1)`
        """
        with torch.no_grad():
            final_next_obs = exp_batch.next_obs[-self.num_envs:]
            _, final_next_state_value = self._network.forward(final_next_obs)
        
        entire_state_value = torch.cat((exp_batch.state_value.detach(), final_next_state_value))
        
        # (num_envs * k, 1) -> (num_envs, k)
        b2e = lambda x: batch2perenv(x, self.num_envs).squeeze_(-1)
        entire_state_value = b2e(entire_state_value)
        reward = b2e(exp_batch.reward)
        terminated = b2e(exp_batch.terminated)
        
        # compute advantage using GAE
        advantage = L.gae(
            entire_state_value,
            reward,
            terminated,
            self._config.gamma,
            self._config.lam
        )
        
        # compute v_target
        target_state_value = advantage + entire_state_value[:, :-1]
        
        # (num_envs, k) -> (num_envs * k, 1)
        e2b = lambda x: perenv2batch(x.unsqueeze_(-1))
        advantage = e2b(advantage)
        target_state_value = e2b(target_state_value)
        
        return advantage, target_state_value

    @property
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/Actor Loss", "Network/Critic Loss")
    
    @property
    def log_data(self) -> dict[str, tuple]:
        ld = super().log_data
        if self.actor_average_loss.count > 0:
            ld["Network/Actor Loss"] = (self.actor_average_loss.mean, self.training_steps)
            ld["Network/Critic Loss"] = (self.critic_average_loss.mean, self.training_steps)
            self.actor_average_loss.reset()
            self.critic_average_loss.reset()
        return ld
