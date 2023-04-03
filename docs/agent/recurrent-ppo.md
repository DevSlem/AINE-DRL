---
sort: 5
---

# Recurrent PPO

**Recurrent PPO** is modified version of [PPO](./ppo.md) to use recurrent network like [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) or [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit). It's useful when you train an agent in [Partially Observable Markov Decision Process (POMDP)](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process). In this setting, the agent can't fully observe the state space. That's why the agent requires action-observation history. The hidden state of recurrent network can achieve this because of memory ability!

PPO is TD method so it's not desirable to work episodic. Our Recurrent PPO uses **Truncated Backpropagation Through Time (T-BPTT)**. The T-BPTT is a technique to train recurrent neural network by truncating the backpropagation through time. The truncated length is called **sequence length**. The truncated length is usually set to `8` or greater value. Therefore, our algorithm can also work non-episodic.

## Configuration

|Parameter|Description|
|:---|:---|
|`n_steps`|(`int`) The number of time steps to collect experiences until training. The number of total experiences (`entire_batch_size`) is `num_envs * n_steps`. Since PPO is on-policy method, the experiences are discarded after training.|
|`epoch`|(`int`) The number of times the entire experience batch is used to update parameters|
|`seq_len`|(`int`) The sequence length of the experience sequence batch when **training**. The entire experience batch is split by `seq_len` unit then results in the experience sequences with `padding_value`. This is why the entire sequence batch size (`entire_seq_batch_size`) is greater than `entire_batch_size`. Typically `8` or greater value are recommended.|
|`seq_mini_batch_size`|(`int`) The sequence mini-batches are selected randomly and independently from the entire experience sequence batch during one epoch. The number of parameters updates at each epoch is the integer value of `entire_seq_batch_size` / `seq_mini_batch_size`.|
|`padding_value`|(`float`, default = `0.0`) Pad sequences to the value for the same `seq_len`.|
|`gamma`|(`float`, default = `0.99`) Discount factor $$\gamma$$ of future rewards.|
|`lam`|(`float`, default = `0.95`) Regularization parameter $$\lambda$$ which controls the bias-variance trade-off of Generalized Advantage Estimation (GAE).|
|`epsilon_clip`|(`float`, default = `0.2`) Clamps the probability ratio ($$\dfrac{\pi_{\text{new}}}{\pi_{\text{old}}}$$) into the range $$[1 - \epsilon, 1 + \epsilon]$$.|
|`value_loss_coef`|(`float`, default = `0.5`) State value loss (critic loss) multiplier.|
|`entropy_coef`|(`float`, default = `0.001`) Entropy multiplier used to compute loss. It adjusts exploration-exploitation trade-off.|
|`device`|(`str | None`, default = `None`) Device on which the agent works. If this setting is `None`, the agent device is same as your network's one. Otherwise, the network device changes to this device. <br><br> Options: `None`, `cpu`, `cuda`, `cuda:0` and other devices of `torch.device()` argument|

## Network

class: `RecurrentPPOSharedNetwork`

Since it uses the recurrent network, you must consider the hidden state which can achieve the action-observation history.

Note that since it uses the Actor-Critic architecture and the parameter sharing, the encoding layer must be shared between Actor and Critic.

You need to implement below methods.

### Forward

```python
@abstractmethod
def forward(
    self, 
    obs_seq: Observation, 
    hidden_state: Tensor
) -> tuple[PolicyDist, Tensor, Tensor]
```

|Input|Description|Shape|
|---|---|---|
|obs_seq (`Observation`)|observation sequence batch tuple|`*batch_shape` = `(seq_batch_size, seq_len)` details in `Observation` docs|
|hidden_state (`Tensor`)|hidden states at the beginning of each sequence|`(D x num_layers, seq_batch_size, H)`|

|Output|Description|Shape|
|---|---|---|
|policy_dist_seq (`PolicyDist`)|policy distribution $$\pi(a \vert s)$$ sequences|`*batch_shape` = `(seq_batch_size, seq_len)` details in `PolicyDist` docs|
|state_value_seq (`Tensor`)|state value $$V(s)$$ sequences|`(seq_batch_size, seq_len, 1)`|
|next_seq_hidden_state (`Tensor`)|hidden states which will be used for the next sequence|`(D x num_layers, seq_batch_size, H)`|

Refer to the following explanation:
        
* `seq_batch_size`: the number of independent sequences
* `seq_len`: the length of each sequence
* `num_layers`: the number of recurrent layers
* `D`: 2 if bidirectional otherwise 1
* `H`: the value depends on the type of the recurrent network

When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pyorg/docs/stable/generated/nn.LSTM.html. <br>
When you use GRU, `H` = `H_out`. See details in https://pyorg/docs/stable/generated/nn.GRU.html.