---
sort: 6
---

# PPO RND

**PPO RND** is combined version of [PPO](./ppo.md) and **Random Network Distillation (RND)**. RND is a method to generate intrinsic reward by prediction error of next state feature. If state space is huge or reward setting is sparse, the agent should be able to explore enough to find better policy. The intrinsic reward can help train the agent to explore the environment.

Paper: [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)

## Configuration

|Parameter|Description|
|:---|:---|
|`n_steps`|(`int`) The number of time steps to collect experiences until training. The number of total experiences (`entire_batch_size`) is `num_envs * n_steps`. Since PPO is on-policy method, the experiences are discarded after training.|
|`epoch`|(`int`) The number of times the entire experience batch is used to update parameters|
|`mini_batch_size`|(`int`) The mini-batches are selected randomly and independently from the entire experience batch during one epoch. The number of parameters updates at each epoch is the integer value of `entire_batch_size` / `mini_batch_size`.|
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

## Network

class: `PPORNDNetwork`

Note that since it uses the Actor-Critic architecture and the parameter sharing, the encoding layer must be shared between Actor and Critic.

RND uses extrinsic and intrinsic reward streams. Each stream can be different episodic or non-episodic, and can have different discount factors. RND constitutes of the predictor and target networks. Both of them should have the similar architectures (not must same) but their initial parameters should not be the same. The target network is deterministic, which means it will be never updated. 

You need to implement below methods.

### Forward Actor-Critic

```python
@abstractmethod
def forward_actor_critic(
    self, 
    obs: Observation
) -> tuple[PolicyDist, Tensor, Tensor]
```

|Input|Description|Shape|
|---|---|---|
|obs (`Observation`)|observation batch tuple|`*batch_shape` = `(batch_size,)` details in `Observation` docs|

|Output|Description|Shape|
|---|---|---|
|policy_dist (`PolicyDist`)|policy distribution $\pi(a \vert s)$|`*batch_shape` = `(batch_size,)` details in `PolicyDist` docs|
|ext_state_value (`Tensor`)|extrinsic state value $V_E(s)$|`(batch_size, 1)`|
|int_state_value (`Tensor`)|intrinsic state value $V_I(s)$|`(batch_size, 1)`|

### Forward RND

```python
@abstractmethod
def forward_rnd(
    self, 
    obs: Observation, 
) -> tuple[Tensor, Tensor]
```

The value of `out_features` depends on you.

|Input|Description|Shape|
|---|---|---|
|obs (`Observation`)|observation batch tuple|`*batch_shape` = `(batch_size,)` details in `Observation` docs|

|Output|Description|Shape|
|---|---|---|
|predicted_feature (`Tensor`)|predicted feature $\hat{f}(s)$ whose gradient flows|`(batch_size, out_features)`|
|target_feature (`Tensor`)|target feature $f(s)$ whose gradient doesn't flow|`(batch_size, out_features)`|