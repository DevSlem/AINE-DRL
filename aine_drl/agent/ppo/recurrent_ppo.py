import torch

import aine_drl.rl_loss as L
import aine_drl.util as util
from aine_drl.agent.agent import Agent, BehaviorType
from aine_drl.agent.ppo.config import RecurrentPPOConfig
from aine_drl.agent.ppo.net import RecurrentPPOSharedNetwork
from aine_drl.agent.ppo.trajectory import (RecurrentPPOExperience,
                                           RecurrentPPOTrajectory)
from aine_drl.exp import Action, Experience, Observation
from aine_drl.net import NetworkTypeError, Trainer
from aine_drl.util.func import batch2perenv, perenv2batch


class RecurrentPPO(Agent):
    """
    Recurrent Proximal Policy Optimization (PPO).
    
    Paper: https://arxiv.org/abs/1707.06347
    """
    def __init__(
        self, 
        config: RecurrentPPOConfig,
        network: RecurrentPPOSharedNetwork,
        trainer: Trainer,
        num_envs: int,
        behavior_type: BehaviorType = BehaviorType.TRAIN,
    ) -> None:        
        if not isinstance(network, RecurrentPPOSharedNetwork):
            raise NetworkTypeError(RecurrentPPOSharedNetwork)
        
        super().__init__(num_envs, network, config.device, behavior_type)
        
        self._config = config
        self._network = network
        self._trainer = trainer
        self._trajectory = RecurrentPPOTrajectory(self._config.n_steps)
        
        self._action_log_prob: torch.Tensor = None # type: ignore
        self._state_value: torch.Tensor = None # type: ignore
        hidden_state_shape = (network.hidden_state_shape()[0], self._num_envs, network.hidden_state_shape()[1])
        self._hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._next_hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._prev_terminated = torch.zeros((self._num_envs, 1), device=self.device)
        
        self._actor_loss_mean = util.IncrementalMean()
        self._critic_loss_mean = util.IncrementalMean()
        
        # for inference mode
        infer_hidden_state_shape = (network.hidden_state_shape()[0], 1, network.hidden_state_shape()[1])
        self._infer_hidden_state = torch.zeros(infer_hidden_state_shape, device=self.device)
        self._infer_next_hidden_state = torch.zeros(infer_hidden_state_shape, device=self.device)
        self._infer_prev_terminated = torch.zeros((1, 1), device=self.device)
        
    @property
    def name(self) -> str:
        return "Recurrent PPO"
    
    @property
    def config_dict(self) -> dict:
        return self._config.__dict__
    
    def _update_train(self, exp: Experience):
        self._prev_terminated = exp.terminated
        
        self._trajectory.add(RecurrentPPOExperience(
            **exp.__dict__,
            action_log_prob=self._action_log_prob,
            state_value=self._state_value,
            hidden_state=self._hidden_state
        ))
        
        if self._trajectory.reached_n_steps:
            self._train()
    
    def _update_inference(self, exp: Experience):
        self._infer_prev_terminated = exp.terminated
    
    @torch.no_grad()
    def _select_action_train(self, obs: Observation) -> Action:
        self._hidden_state = self._next_hidden_state * (1.0 - self._prev_terminated)
        
        # feed forward
        # when interacting with environment, sequence length must be 1
        # *batch_shape = (seq_batch_size, seq_len) = (num_envs, 1)
        policy_dist_seq, state_value_seq, next_hidden_state = self._network.forward(
            obs.transform(lambda o: o.unsqueeze(dim=1)),
            self._hidden_state
        )
        
        # action sampling
        action_seq = policy_dist_seq.sample()
        
        # (num_envs, 1, *shape) -> (num_envs, *shape)
        action = action_seq.transform(lambda a: a.squeeze(dim=1))
        self._action_log_prob = policy_dist_seq.joint_log_prob(action_seq).squeeze_(dim=1)
        self._state_value = state_value_seq.squeeze_(dim=1)
        
        self._next_hidden_state = next_hidden_state
        
        return action
    
    @torch.no_grad()
    def _select_action_inference(self, obs: Observation) -> Action:
        self._infer_hidden_state = self._infer_next_hidden_state * (1.0 - self._infer_prev_terminated)
        policy_dist_seq, _, next_hidden_state = self._network.forward(
            obs.transform(lambda o: o.unsqueeze(dim=1)),
            self._infer_hidden_state
        )
        action_seq = policy_dist_seq.sample()
        self._infer_next_hidden_state = next_hidden_state
        return action_seq.transform(lambda a: a.squeeze(dim=1))
            
    def _train(self):
        # sample experience batch from the trajectory
        exp_batch = self._trajectory.sample()
        
        # compute advantage and target state value
        advantage, target_state_value = self._compute_adv_target(exp_batch)
        
        # convert batch to truncated sequence
        seq_generator = util.TruncatedSeqGen(
            self._config.seq_len, 
            self._num_envs, 
            self._config.n_steps, 
            self._config.padding_value
        )
        
        def add_to_seq_gen(batch, start_idx = 0, seq_len = 0):
            seq_generator.add(batch2perenv(batch, self._num_envs), start_idx=start_idx, seq_len=seq_len)
            
        add_to_seq_gen(exp_batch.hidden_state.swapaxes(0, 1), seq_len=1)
        for obs_tensor in exp_batch.obs.items:
            add_to_seq_gen(obs_tensor)
        if exp_batch.action.num_discrete_branches > 0:
            add_to_seq_gen(exp_batch.action.discrete_action)
        else:
            seq_generator.add(torch.empty((self._num_envs, self._config.n_steps, 0)))
        if exp_batch.action.num_continuous_branches > 0:
            add_to_seq_gen(exp_batch.action.continuous_action)
        else:
            seq_generator.add(torch.empty((self._num_envs, self._config.n_steps, 0)))
        add_to_seq_gen(exp_batch.action_log_prob)
        add_to_seq_gen(advantage)
        add_to_seq_gen(target_state_value)
        
        sequences = seq_generator.generate(batch2perenv(exp_batch.terminated, self._num_envs).unsqueeze_(-1))
        mask = sequences[0]
        seq_init_hidden_state = sequences[1]
        obs_seq = Observation(sequences[2:2 + exp_batch.obs.num_items])
        discrete_action_seq, continuous_action_seq, old_action_log_prob_seq, advantage_seq, target_state_value_seq = sequences[2 + exp_batch.obs.num_items:]
        
        num_seq = len(mask)
        # (num_seq, 1, D x num_layers, H) -> (D x num_layers, num_seq, H)
        seq_init_hidden_state = seq_init_hidden_state.squeeze_(dim=1).swapaxes_(0, 1)
        
        for _ in range(self._config.epoch):
            shuffled_seq_batch_idx = torch.randperm(num_seq)
            for i in range(num_seq // self._config.seq_mini_batch_size):
                # when sliced by sample_seq_idx, (entire_seq_batch_size,) -> (seq_mini_batch_size,)
                sample_seq_idx = shuffled_seq_batch_idx[self._config.seq_mini_batch_size * i : self._config.seq_mini_batch_size * (i + 1)]
                # when masked by sample_mask, (seq_mini_batch_size, seq_len) -> (masked_batch_size,)
                sample_mask = mask[sample_seq_idx]
                
                # feed forward
                sample_policy_dist_seq, sample_state_value_seq, _ = self._network.forward(
                    obs_seq[sample_seq_idx], 
                    seq_init_hidden_state[:, sample_seq_idx]
                )
                
                # compute actor loss
                sample_action_seq = Action(
                    discrete_action_seq[sample_seq_idx],
                    continuous_action_seq[sample_seq_idx]
                )
                sample_new_action_log_prob_seq = sample_policy_dist_seq.joint_log_prob(sample_action_seq)
                actor_loss = L.ppo_clipped_loss(
                    advantage_seq[sample_seq_idx][sample_mask],
                    old_action_log_prob_seq[sample_seq_idx][sample_mask],
                    sample_new_action_log_prob_seq[sample_mask],
                    self._config.epsilon_clip
                )
                entropy = sample_policy_dist_seq.joint_entropy()[sample_mask].mean()
                
                # compute critic loss
                critic_loss = L.bellman_value_loss(
                    sample_state_value_seq[sample_mask], 
                    target_state_value_seq[sample_seq_idx][sample_mask]
                )
                
                # train step
                loss = actor_loss + self._config.value_loss_coef * critic_loss - self._config.entropy_coef * entropy
                self._trainer.step(loss, self.training_steps)
                self._tick_training_steps()
                
                # update log data
                self._actor_loss_mean.update(actor_loss.item())
                self._critic_loss_mean.update(critic_loss.item())

    def _compute_adv_target(self, exp_batch: RecurrentPPOExperience) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantage, v_target.

        Args:
            exp_batch (PPOExperienceBatch): experience batch

        Returns:
            advantage (Tensor): `(num_envs x n_steps, 1)`
            v_target (Tensor): `(num_envs x n_steps, 1)`
        """
        
        # (num_envs, *obs_shape)
        final_next_obs = exp_batch.next_obs[-self._num_envs:]
        final_next_hidden_state = self._next_hidden_state.to(device=self.device)
        
        # feed forward without gradient calculation
        with torch.no_grad():
            # (num_envs, 1, *obs_shape) because sequence length is 1
            _, final_next_state_value_seq, _ = self._network.forward(
                final_next_obs.transform(lambda o: o.unsqueeze(dim=1)),
                final_next_hidden_state
            )
        
        # (num_envs, 1, 1) -> (num_envs, 1)
        final_next_state_value = final_next_state_value_seq.squeeze_(dim=1)
        # (num_envs x (n_steps + 1), 1)
        entire_state_value = torch.cat((exp_batch.state_value, final_next_state_value))
        
        # (num_envs x k, 1) -> (num_envs, k, 1) -> (num_envs, k)
        b2e = lambda x: batch2perenv(x, self._num_envs).squeeze_(dim=-1)
        entire_state_value = b2e(entire_state_value)
        reward = b2e(exp_batch.reward)
        terminated = b2e(exp_batch.terminated)
        
        # compute advantage (num_envs, n_steps) using GAE
        advantage = L.gae(
            entire_state_value,
            reward,
            terminated,
            self._config.gamma,
            self._config.lam
        )
        
        # compute target state_value (num_envs, n_steps)
        target_state_value = advantage + entire_state_value[:, :-1]
        
        # (num_envs, n_steps) -> (num_envs x n_steps, 1)
        e2b = lambda x: perenv2batch(x.unsqueeze_(dim=-1))
        advantage = e2b(advantage)
        target_state_value = e2b(target_state_value)
        
        return advantage, target_state_value

    @property
    def log_keys(self) -> tuple[str, ...]:
        return super().log_keys + ("Network/Actor Loss", "Network/Critic Loss")
    
    @property
    def log_data(self) -> dict[str, tuple]:
        ld = super().log_data
        if self._actor_loss_mean.count > 0:
            ld["Network/Actor Loss"] = (self._actor_loss_mean.mean, self.training_steps)
            ld["Network/Critic Loss"] = (self._critic_loss_mean.mean, self.training_steps)
            self._actor_loss_mean.reset()
            self._critic_loss_mean.reset()
        return ld
