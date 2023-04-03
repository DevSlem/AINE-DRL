---
sort: 7
---

# Recurrent PPO RND

**Recurrent PPO RND** is modified version of [PPO RND](./ppo-rnd.md) to use recurrent network like [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) or [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit). You can see why recurrent network is useful in [Recurrent PPO](./recurrent-ppo.md).

## Configuration

|Parameter|Description|
|:---|:---|
|`n_steps`|(`int`) The number of time steps to collect experiences until training. The number of total experiences (`entire_batch_size`) is `num_envs * n_steps`. Since PPO is on-policy method, the experiences are discarded after training.|
|`epoch`|(`int`) The number of times the entire experience batch is used to update parameters|
|`seq_len`|(`int`) The sequence length of the experience sequence batches when **training**. T1he entire experience batch is split by `seq_len` unit to the experience sequence batches with `padding_value`. This is why the entire sequence batch size (`entire_seq_batch_size`) is greater than `entire_batch_size`. Typically `8` or greater value are recommended.|
|`seq_mini_batch_size`|(`int`) The sequence mini-batches are selected randomly and independently from the entire experience sequence batch during one epoch. The number of parameters updates at each epoch is the integer value of `entire_seq_batch_size` / `seq_mini_batch_size`.|
|`padding_value`|(`float`, default = `0.0`) Pad sequences to the value for the same `seq_len`.|
|`ext_gamma`|(`float`, default = `0.999`) Discount factor $$\gamma_E$$ of future extrinsic rewards.|
|`int_gamma`|(`float`, default = `0.99`) Discount factor $$\gamma_I$$ of future intrinsic rewards.|
|`ext_adv_coef`|(`float`, default = `1.0`) Extrinsic advantage multiplier.|
|`int_adv_coef`|(`float`, default = `1.0`) Intrinsic advantage multiplier.|
|`lam`|(`float`, default = `0.95`) Regularization parameter $$\lambda$$ which controls the bias-variance trade-off of Generalized Advantage Estimation (GAE).|
|`epsilon_clip`|(`float`, default = `0.2`) Clamps the probability ratio ($$\dfrac{\pi_{\text{new}}}{\pi_{\text{old}}}$$) into the range $$[1 - \epsilon, 1 + \epsilon]$$.|
|`value_loss_coef`|(`float`, default = `0.5`) State value loss (critic loss) multiplier.|
|`entropy_coef`|(`float`, default = `0.001`) Entropy multiplier used to compute loss. It adjusts exploration-exploitation trade-off.|
|`rnd_pred_exp_proportion`|(`float`, default = `0.25`) The proportion of experiences used to train RND predictor to keep the effective batch size.|
|`init_norm_steps`|(`int | None`, default = `50`) The initial time steps to initialize normalization parameters of both observation and hidden state. When the value is `None`, the algorithm never normalize them during training.|
|`obs_norm_clip_range`|([`float`, `float`], default = [`-5.0`, `5.0`]) Clamps the normalized observation into the range [`min`, `max`].|
|`hidden_state_norm_clip_range`|([`float`, `float`], default = [`-5.0`, `5.0`]) Clamps the normalized hidden state into the range [`min`, `max`].|
|`device`|(`str | None`, default = `None`) Device on which the agent works. If this setting is `None`, the agent device is same as your network's one. Otherwise, the network device changes to this device. <br><br> Options: `None`, `cpu`, `cuda`, `cuda:0` and other devices of `torch.device()` argument|

## Network

class: `RecurrentPPORNDNetwork`

Since it uses the recurrent network, you must consider the hidden state which can achieve the action-observation history.

Note that since PPO uses the Actor-Critic architecure and the parameter sharing, the encoding layer must be shared between Actor and Critic. Be careful not to share parameters between PPO and RND networks.

RND uses extrinsic and intrinsic reward streams. Each stream can be different episodic or non-episodic, and can have different discount factors. RND constitutes of the predictor and target networks. Both of them should have the similar architectures (not must same) but their initial parameters should not be the same. The target network is deterministic, which means it will be never updated. 

You need to implement below methods.

### Forward Actor-Critic

```python
@abstractmethod
def forward_actor_critic(
    self, 
    obs_seq: Observation, 
    hidden_state: Tensor
) -> tuple[PolicyDist, Tensor, Tensor, Tensor]
```

|Input|Description|Shape|
|---|---|---|
|obs_seq (`Observation`)|observation sequence batch tuple|`*batch_shape` = `(seq_batch_size, seq_len)` details in `Observation` docs|
|hidden_state (`Tensor`)|hidden states at the beginning of each sequence|`(D x num_layers, seq_batch_size, H)`|

|Output|Description|Shape|
|---|---|---|
|policy_dist_seq (`PolicyDist`)|policy distribution $$\pi(a \vert s)$$ sequences|`*batch_shape` = `(seq_batch_size, seq_len)` details in `PolicyDist` docs|
|ext_state_value_seq (`Tensor`)|extrinsic state value $$V_E(s)$$ sequences|`(seq_batch_size, seq_len, 1)`|
|int_state_value_seq (`Tensor`)|intrinsic state value $$V_I(s)$$ sequences|`(seq_batch-size, seq_len, 1)`|
|next_seq_hidden_state (`Tensor`)|hidden states which will be used for the next sequence|`(D x num_layers, seq_batch_size, H)`|

Refer to the following explanation:
        
* `seq_batch_size`: the number of independent sequences
* `seq_len`: the length of each sequence
* `num_layers`: the number of recurrent layers
* `D`: 2 if bidirectional otherwise 1
* `H`: the value depends on the type of the recurrent network

When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. <br>
When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.

### Forward RND

```python
@abstractmethod
def forward_rnd(
    self, 
    obs: Observation, 
    hidden_state: Tensor
) -> tuple[Tensor, Tensor]
```

The value of `out_features` depends on you.

|Input|Description|Shape|
|---|---|---|
|obs (`Observation`)|observation batch tuple|`*batch_shape` = `(batch_size,)` details in `Observation` docs|
|hidden_state (`Tensor`)|hidden state batch with flattened features|`(batch_size, D x num_layers x H)`|

|Output|Description|Shape|
|---|---|---|
|predicted_feature (`Tensor`)|predicted feature $$\hat{f}(s)$$ whose gradient flows|`(batch_size, out_features)`|
|target_feature (`Tensor`)|target feature $$f(s)$$ whose gradient doesn't flow|`(batch_size, out_features)`|