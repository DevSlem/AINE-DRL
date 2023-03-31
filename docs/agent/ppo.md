# PPO

**Proximal Policy Optimization (PPO)** is one of the most powerful actor-critic methods. It can stably update policy parameters in trust region using surrogate objective function.

PPO suggests two objective functions. We use clipped surrogate objective function of them, which is known to have better performance.

Paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

PPO has below features:

* Actor-Critic method
* on-policy
* stable
* general

## Configuration

PPO has slightly complex hyperparameters, but default values are good enough.

> Note if the setting has default value, you can skip it.

|Parameter|Description|
|:---|:---|
|`n_steps`|(`int`) The number of time steps to collect experiences until training. The number of total experiences (`entire_batch_size`) is `num_envs * n_steps`. Since PPO is on-policy method, the experiences are discarded after training.|
|`epoch`|(`int`) The number of times the entire experience batch is used to update parameters|
|`mini_batch_size`|(`int`) The mini-batches are selected randomly and independently from the entire experience batch during one epoch. The number of parameters updates at each epoch is the integer value of `entire_batch_size` / `mini_batch_size`.|
|`gamma`|(`float`, default = `0.99`) Discount factor $\gamma$ of future rewards.|
|`lam`|(`float`, default = `0.95`) Regularization parameter $\lambda$ which controls the bias-variance trade-off of Generalized Advantage Estimation (GAE).|
|`advantage_normalization`|(`bool`, default = `False`) Whether or not normalize advantage estimates across single mini batch. It may reduce variance and lead to stability, but does not seem to effect performance much.|
|`epsilon_clip`|(`float`, default = `0.2`) Clamps the probability ratio ($\dfrac{\pi_{\text{new}}}{\pi_{\text{old}}}$) into the range $[1 - \epsilon, 1 + \epsilon]$.|
|`value_loss_coef`|(`float`, default = `0.5`) State value loss (critic loss) multiplier.|
|`entropy_coef`|(`float`, default = `0.001`) Entropy multiplier used to compute loss. It adjusts exploration-exploitation trade-off.|

## Network

class: `PPOSharedNetwork`

Note that since it uses the Actor-Critic architecure and the parameter sharing, the encoding layer must be shared between Actor and Critic.

### Forward

|Input|Description|Shape|
|---|---|---|
|obs (`Observation`)|observation batch tuple|`*batch_shape` = `(batch_size,)` details in `Observation` docs|

|Output|Description|Shape|
|---|---|---|
|policy_dist (`PolicyDist`)|policy distribution|`*batch_shape` = `(batch_size,)` details in `PolicyDist` docs|
|state_value (`Tensor`)|state value $V(s)$|`(batch_size, 1)`|