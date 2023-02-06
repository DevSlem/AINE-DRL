from typing import Dict, NamedTuple, Optional, Tuple
from aine_drl.agent import Agent
from aine_drl.experience import ActionTensor, Experience
from aine_drl.network import RecurrentActorCriticSharedNetwork
from aine_drl.policy.policy import Policy
from .ppo import PPO
from .ppo_trajectory import RecurrentPPOExperienceBatch, RecurrentPPOTrajectory
import aine_drl.drl_util as drl_util
import aine_drl.util as util
import torch

class RecurrentPPOConfig(NamedTuple):
    """
    Recurrent PPO configurations.

    Args:
        `training_freq (int)`: training frequency which is the number of time steps to gather experiences
        `epoch (int)`: number of using total experiences to update parameters at each training frequency
        `sequence_length (int)`: sequence length of recurrent network when training. trajectory is split by `sequence_length` unit. a value of `8` or greater are typically recommended.
        `num_sequences_per_step (int)`: number of sequences per train step, which are selected randomly
        `padding_value (float, optional)`: pad sequences to the value for the same `sequence_length`. Defaults to 0.
        `gamma (float, optional)`: discount factor. Defaults to 0.99.
        `lam (float, optional)`: regularization parameter which controls the balanace of Generalized Advantage Estimation (GAE) between bias and variance. Defaults to 0.95.
        `epsilon_clip (float, optional)`: clipping the probability ratio (pi_theta / pi_theta_old) to [1-eps, 1+eps]. Defaults to 0.2.
        `value_loss_coef (float, optional)`: state value loss (critic loss) multiplier. Defaults to 0.5.
        `entropy_coef (float, optional)`: entropy multiplier used to compute loss. It adjusts exploration/exploitation balance. Defaults to 0.001.
        `grad_clip_max_norm (float | None, optional)`: maximum norm for the gradient clipping. Defaults to no gradient clipping.
    """
    training_freq: int
    epoch: int
    sequence_length: int
    num_sequences_per_step: int
    padding_value: float = 0.0
    gamma: float = 0.99
    lam: float = 0.95
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    grad_clip_max_norm: Optional[float] = None
    

class RecurrentPPO(Agent):
    """
    Recurrent Proximal Policy Optimization (PPO) using RNN. See details in https://arxiv.org/abs/1707.06347.

    Args:
        config (RecurrentPPOConfig): Recurrent PPO configuration
        network (RecurrentActorCriticNetwork): recurrent actor critic network (e.g., LSTM, GRU)
        policy (Policy): policy
        num_envs (int): number of environments
    """
    def __init__(self, 
                 config: RecurrentPPOConfig,
                 network: RecurrentActorCriticSharedNetwork,
                 policy: Policy,
                 num_envs: int) -> None:        
        if not isinstance(network, RecurrentActorCriticSharedNetwork):
            raise TypeError("The network type must be RecurrentActorCriticSharedNetwork.")
        
        super().__init__(network, policy, num_envs)
        
        self.config = config
        self.network = network
        self.trajectory = RecurrentPPOTrajectory(self.config.training_freq)
        
        self.current_action_log_prob = None
        self.v_pred = None
        hidden_state_shape = (network.hidden_state_shape[0], self.num_envs, network.hidden_state_shape[1])
        self.current_hidden_state = torch.zeros(hidden_state_shape)
        self.next_hidden_state = torch.zeros(hidden_state_shape)
        self.prev_terminated = torch.zeros(self.num_envs, 1)
        
        self.actor_average_loss = util.IncrementalAverage()
        self.critic_average_loss = util.IncrementalAverage()
        
        # for inference mode
        inference_hidden_state_shape = (network.hidden_state_shape[0], 1, network.hidden_state_shape[1])
        self.inference_current_hidden_state = torch.zeros(inference_hidden_state_shape)
        self.inference_next_hidden_state = torch.zeros(inference_hidden_state_shape)
        self.inference_prev_terminated = torch.zeros(1, 1)
        
    def update(self, experience: Experience):
        super().update(experience)
        
        self.prev_terminated = torch.from_numpy(experience.terminated)
        
        # add the experience
        self.trajectory.add(
            experience,
            self.current_action_log_prob,
            self.v_pred,
            self.current_hidden_state
        )
        
        # if training frequency is reached, start training
        if self.trajectory.count == self.config.training_freq:
            self._train()
            
    def inference(self, experience: Experience):
        self.inference_prev_terminated = torch.from_numpy(experience.terminated)
            
    def select_action_train(self, obs: torch.Tensor) -> ActionTensor:
        with torch.no_grad():
            self.current_hidden_state = self.next_hidden_state * (1.0 - self.prev_terminated)
            # when interacting with environment, sequence_length must be 1
            # feed forward
            pdparam_seq, state_value_seq, next_hidden_state = self.network.forward(obs.unsqueeze(dim=1), self.current_hidden_state.to(device=self.device))
            
            # action sampling
            pdparam = pdparam_seq.sequence_to_flattened()
            dist = self.policy.get_policy_distribution(pdparam)
            action = dist.sample()
            
            # store data
            self.current_action_log_prob = dist.log_prob(action).cpu()
            self.v_pred = state_value_seq.squeeze_(dim=1).cpu()
            self.next_hidden_state = next_hidden_state.cpu()
            
            return action
    
    def select_action_inference(self, obs: torch.Tensor) -> ActionTensor:
        self.inference_current_hidden_state = self.inference_next_hidden_state * (1.0 - self.inference_prev_terminated)
        pdparam, _, hidden_state = self.network.forward(obs.unsqueeze(1), self.inference_current_hidden_state.to(device=self.device))
        dist = self.policy.get_policy_distribution(pdparam.sequence_to_flattened())
        action = dist.sample()
        self.inference_next_hidden_state = hidden_state.cpu()
        return action
            
    def _train(self):
        # sample experience batch from the trajectory
        exp_batch = self.trajectory.sample(self.device)
        
        # compute advantage and target state value
        advantage, target_state_value = self._compute_adavantage_target_state_value(exp_batch)
        
        # convert batch to truncated sequence
        seq_generator = drl_util.TruncatedSequenceGenerator(self.config.sequence_length, self.num_envs, exp_batch.n_steps, self.config.padding_value)
        
        def add_to_seq_gen(batch, start_idx = 0, seq_len = 0):
            seq_generator.add(drl_util.batch2perenv(batch, self.num_envs), start_idx=start_idx, seq_len=seq_len)
            
        add_to_seq_gen(exp_batch.hidden_state.swapaxes(0, 1), seq_len=1)
        add_to_seq_gen(exp_batch.obs)
        if exp_batch.action.num_discrete_branches > 0:
            add_to_seq_gen(exp_batch.action.discrete_action)
        else:
            seq_generator.add(torch.empty((self.num_envs, exp_batch.n_steps, 0)))
        if exp_batch.action.num_continuous_branches > 0:
            add_to_seq_gen(exp_batch.action.continuous_action)
        else:
            seq_generator.add(torch.empty((self.num_envs, exp_batch.n_steps, 0)))
        add_to_seq_gen(exp_batch.action_log_prob)
        add_to_seq_gen(advantage)
        add_to_seq_gen(target_state_value)
        
        sequences = seq_generator.generate(drl_util.batch2perenv(exp_batch.terminated, self.num_envs).unsqueeze_(-1))
        mask, seq_init_hidden_state, obs_seq, discrete_action_seq, continuous_action_seq, old_action_log_prob_seq, advantage_seq, target_state_value_seq = sequences
        
        num_seq = len(mask)
        # (num_seq, 1, D x num_layers, H) -> (D x num_layers, num_seq, H)
        seq_init_hidden_state = seq_init_hidden_state.squeeze_(dim=1).swapaxes_(0, 1)
        
        for _ in range(self.config.epoch):
            sample_sequences = torch.randperm(num_seq)
            for i in range(num_seq // self.config.num_sequences_per_step):
                # when sliced by sample_seq, (num_seq,) -> (num_seq_per_step,)
                sample_seq = sample_sequences[self.config.num_sequences_per_step * i : self.config.num_sequences_per_step * (i + 1)]
                # when masked by m, (num_seq_per_step, seq_len) -> (masked_batch_size,)
                m = mask[sample_seq]
                
                # feed forward
                # in this case num_seq = num_seq_per_step
                pdparam_seq, state_value_seq, _ = self.network.forward(obs_seq[sample_seq], seq_init_hidden_state[:, sample_seq])
                
                # compute actor loss
                # (num_seq_per_step, seq_len, *pdparam_shape) -> (num_seq_per_step * seq_len, *pdparam_shape)
                pdparam = pdparam_seq.sequence_to_flattened()
                dist = self.policy.get_policy_distribution(pdparam)
                # (num_seq_per_step, seq_len, num_actions) -> (num_seq_per_step * seq_len, num_actions)
                a = ActionTensor(
                    discrete_action_seq[sample_seq].flatten(0, 1),
                    continuous_action_seq[sample_seq].flatten(0, 1)
                )
                # (num_seq_per_step * seq_len, 1) -> (num_seq_per_step, seq_len, 1)
                new_action_log_prob_seq = dist.log_prob(a).reshape(self.config.num_sequences_per_step, -1, 1)
                actor_loss = PPO.compute_actor_loss(
                    advantage_seq[sample_seq][m],
                    old_action_log_prob_seq[sample_seq][m],
                    new_action_log_prob_seq[m],
                    self.config.epsilon_clip
                )
                entropy = dist.entropy().reshape(self.config.num_sequences_per_step, -1, 1)[m].mean()
                
                # compute critic loss
                critic_loss = PPO.compute_critic_loss(state_value_seq[m], target_state_value_seq[sample_seq][m])
                
                # train step
                loss = actor_loss + self.config.value_loss_coef * critic_loss - self.config.entropy_coef * entropy
                self.network.train_step(loss, self.config.grad_clip_max_norm, self.clock.training_step)
                self.clock.tick_training_step()
                
                # log data
                self.actor_average_loss.update(actor_loss.item())
                self.critic_average_loss.update(critic_loss.item())

    def _compute_adavantage_target_state_value(self, exp_batch: RecurrentPPOExperienceBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantage, v_target.

        Args:
            exp_batch (PPOExperienceBatch): experience batch

        Returns:
            advantage (Tensor): `(num_envs x n_steps, 1)`
            v_target (Tensor): `(num_envs x n_steps, 1)`
        """
        
        # (num_envs, *obs_shape)
        final_next_obs = exp_batch.next_obs[-self.num_envs:]
        final_next_hidden_state = self.next_hidden_state.to(device=self.device)
        
        # feed forward without gradient calculation
        with torch.no_grad():
            # (num_envs, 1, *obs_shape) because sequence length is 1
            _, final_next_state_value_seq, _ = self.network.forward(final_next_obs.unsqueeze(dim=1), final_next_hidden_state)
        
        # (num_envs, 1, 1) -> (num_envs, 1)
        final_next_state_value = final_next_state_value_seq.squeeze_(dim=1)
        # (num_envs x (n_steps + 1), 1)
        total_state_value = torch.cat([exp_batch.v_pred, final_next_state_value])
        
        # (num_envs x T, 1) -> (num_envs, T, 1) -> (num_envs, T)
        total_state_value = drl_util.batch2perenv(total_state_value, self.num_envs).squeeze_(-1) # T = n_steps + 1
        reward = drl_util.batch2perenv(exp_batch.reward, self.num_envs).squeeze_(-1) # T = n_steps
        terminated = drl_util.batch2perenv(exp_batch.terminated, self.num_envs).squeeze_(-1) # T = n_steps
        
        # compute advantage (num_envs, n_steps) using GAE
        advantage = drl_util.compute_gae(
            total_state_value,
            reward,
            terminated,
            self.config.gamma,
            self.config.lam
        )
        
        # compute target state_value (num_envs, n_steps)
        target_state_value = advantage + total_state_value[:, :-1]
        
        # (num_envs, n_steps) -> (num_envs x n_steps, 1)
        advantage = drl_util.perenv2batch(advantage.unsqueeze_(-1))
        target_state_value = drl_util.perenv2batch(target_state_value.unsqueeze_(-1))
        
        return advantage, target_state_value

    @property
    def log_keys(self) -> Tuple[str, ...]:
        return super().log_keys + ("Network/Actor Loss", "Network/Critic Loss")
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        ld = super().log_data
        if self.actor_average_loss.count > 0:
            ld["Network/Actor Loss"] = (self.actor_average_loss.average, self.clock.training_step)
            ld["Network/Critic Loss"] = (self.critic_average_loss.average, self.clock.training_step)
            self.actor_average_loss.reset()
            self.critic_average_loss.reset()
        return ld
