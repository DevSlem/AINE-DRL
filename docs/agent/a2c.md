---
sort: 2
---

# A2C

**Advantage Actor-Critic (A2C)** is a simple actor-critic method. Differnt from [REINFORCE](https://github.com/DevSlem/AINE-DRL/wiki/REINFORCE), it has a simple idea: instead of computing return $$G_t$$, estimate state value $$V(s)$$ and *bootstrapping*! A2C uses advantage function $$A(s,a) = Q(s,a) - V(s)$$ to update policy parameters. $$Q(s,a)$$ is action value, $$V(s)$$ is state value.

A2C has below features:

* policy-based
* Temporal Difference (TD) method
* on-policy
* high bias, low variance

Since A2C is TD method, you don't need to wait to update parameters until an episode is terminated. You can do online and batch learning.

You can see source code in [a2c](https://github.com/DevSlem/AINE-DRL/tree/main/aine_drl/agent/a2c).

## Configuration

Since it has simple hyperparameters, you don't need to understand deeply reinforcement learning.

> Note if the setting has default value, you can skip it.

|Setting|Description|
|---|---|
|`n_steps`|(`int`) The number of time steps to collect experiences until training. The number of total experiences (`entire_batch_size`) which is used to training is `num_envs` x `n_steps`. Since PPO is on-policy method, the experiences are discarded after training.|
|`gamma`|(`float`, default = `0.99`) Discount factor $$\gamma$$ of future rewards.|
|`lam`|(`float`, default = `0.95`) Regularization parameter $$\lambda$$ which controls the bias-variance trade-off of Generalized Advantage Estimation (GAE).|
|`value_loss_coef`|(`float`, default = `0.5`) State value loss (critic loss) multiplier.|
|`entropy_coef`|(`float`, default = `0.001`) Entropy multiplier used to compute loss. It adjusts exploration/exploitation balance.|

## Network

class: `A2CSharedNetwork`

Note that since it uses the Actor-Critic architecure and the parameter sharing, the encoding layer must be shared between Actor and Critic.

You need to implement below methods.

### Forward

```python
@abstractmethod
def forward(
    self, 
    obs: Observation
) -> tuple[PolicyDist, Tensor]
```

|Input|Description|Shape|
|---|---|---|
|obs (`Observation`)|observation batch tuple|`*batch_shape` = `(batch_size,)` details in `Observation` docs|

|Output|Description|Shape|
|---|---|---|
|policy_dist (`PolicyDist`)|policy distribution $$\pi(a \vert s)$$|`*batch_shape` = `(batch_size,)` details in `PolicyDist` docs|
|state_value (`Tensor`)|state value $$V(s)$$|`(batch_size, 1)`|