# Recurrent PPO RND

## Configuration

|Parameter|Description|
|:---|:---|
|`n_steps`|(`int`) The number of time steps to collect experiences until training. The number of total experiences (`entire_batch_size`) is `num_envs * n_steps`. Since PPO is on-policy method, the experiences are discarded after training.|
|`epoch`|(`int`) The number of times the entire experience batch is used to update parameters|
|`seq_len`|(`int`) The sequence length of the experience sequence batches when **training**. T1he entire experience batch is split by `seq_len` unit to the experience sequence batches with `padding_value`. This is why the entire sequence batch size (`entire_seq_batch_size`) is greater than `entire_batch_size`. Typically `8` or greater value are recommended.|
|`seq_mini_batch_size`|(`int`) The sequence mini-batches are selected randomly and independently from the entire experience sequence batch during one epoch. The number of parameters updates at each epoch is the integer value of `entire_seq_batch_size` / `seq_mini_batch_size`.|
|`padding_value`|(`float`, default = `0.0`) Pad sequences to the value for the same `seq_len`.|
|`ext_gamma`|(`float`, default = `0.999`) Discount factor $\gamma_E$ of future extrinsic rewards.|
|`int_gamma`|(`float`, default = `0.99`) Discount factor $\gamma_I$ of future intrinsic rewards.|
|`ext_adv_coef`|(`float`, default = `1.0`) Extrinsic advantage multiplier.|
|`int_adv_coef`|(`float`, default = `1.0`) Intrinsic advantage multiplier.|
|`lam`|(`float`, default = `0.95`) Regularization parameter $\lambda$ which controls the bias-variance trade-off of Generalized Advantage Estimation (GAE).|
|`epsilon_clip`|(`float`, default = `0.2`) Clamps the probability ratio ($\dfrac{\pi_{\text{new}}}{\pi_{\text{old}}}$) into the range $[1 - \epsilon, 1 + \epsilon]$.|
|`value_loss_coef`|(`float`, default = `0.5`) State value loss (critic loss) multiplier.|
|`entropy_coef`|(`float`, default = `0.001`) Entropy multiplier used to compute loss. It adjusts exploration-exploitation trade-off.|
|`rnd_pred_exp_proportion`|(`float`, default = `0.25`) The proportion of experiences used to train RND predictor to keep the effective batch size.|
|`init_norm_steps`|(`int \| None`, default = `50`) The initial time steps to initialize normalization parameters of both observation and hidden state. When the value is `None`, the algorithm never normalize them during training.|
|`obs_norm_clip_range`|([`float`, `float`], default = [`-5.0`, `5.0`]) Clamps the normalized observation into the range [`min`, `max`].|
|`hidden_state_norm_clip_range`|([`float`, `float`], default = [`-5.0`, `5.0`]) Clamps the normalized hidden state into the range [`min`, `max`].|