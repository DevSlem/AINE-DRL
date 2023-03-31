---
sort: 3
---

# Double DQN

**Double Deep Q Network (DQN)** is a value-based off-policy TD method. It estimates action value \$$Q(s,a)$$ and sample actions from the values using policy (e.g., \$$\epsilon$$-greedy policy). Double DQN is improved version of DQN. It uses Double Q-learning idea in a tabular setting so that it reduces the observed overestimations. 

Paper: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

Double DQN has below features:

* value-based
* Temporal Difference (TD) method
* off-policy
* high bias, low variance

Since DQN is TD method, you don't need to wait to update parameters until an episode is terminated. Also, it uses replay buffer which has fixed buffer size. Replay buffer stores experiences and samples the part of them from the buffer. It's because DQN is off-policy method.

## Configuration

Double DQN is simple but you need to consider carefully some hyperparameters. It may significantly affect the training performance.

> Note if the setting has default value, you can skip it.

|Setting|Description|
|---|---|
|`n_steps`|(`int`) The number of time steps to collect experiences until training. The number of total experiences (`entire_batch_size`) which is used to training is `num_envs` x `n_steps`. Since DQN is off-policy method, the experiences can be reused even if they have been used for training.|
|`batch_size`|(`int`) The size of experience batch from the replay buffer|
|`capacity`|(`int`) The number of experineces to be stored in replay buffer. If it exceeds the capacity, the oldest experience is removed (FIFO).|
|`epoch`|(`int`) The number of parameters updates at each `n_steps`|
|`gamma`|(`float`, default = `0.99`) Discount factor \$$\gamma$$ of future rewards.|
|`replace_freq`|(`int \| None`, default = `None`) The frequency of entirely replacing the target network with the update network. It can stabilize training since the target \$$Q$$ value is fixed. |
|`polyak_ratio`|(`float \| None`, default = `None`) The target network is weighted replaced with the update network. The higher the value, the more replaced with the update network parameters. The value \$$\tau$$ must be \$$0 < \tau \leq 1$$.|
|`replay_buffer_device`|(`str`, default = `"auto"`) What device the replay buffer uses. Since replay buffer may use a lot of memory space, you need to consider which device to store the experiences on. Default is network device. <br><br> Options: `auto`, `cpu`, `cuda`, `cuda:0` and etc|

If both `replace_freq` and `polyak_ratio` are `None`, it uses `replace_freq` as `1`. If both of them are set any value, it uses `replace_freq`.

## Network

class: `DoubleDQNNetwork`:

Note that policy distribution according to the action value is allowed (e.g., \$$\epsilon$$-greedy policy, Boltzmann policy).

You need to implement below methods.

### Forward

```python
@abstractmethod
def forward(
    self, 
    obs: Observation
) -> tuple[PolicyDist, ActionValue]
```

|Input|Description|Shape|
|---|---|---|
|obs (`Observation`)|observation batch tuple|`*batch_shape` = `(batch_size,)` details in `Observation` docs|

|Output|Description|Shape|
|---|---|---|
|policy_dist (`PolicyDist`)|policy distribution \$$\pi(a \vert s)$$|`*batch_shape` = `(batch_size,)` details in `PolicyDist` docs|
|action_value (`ActionValue`)|action value \$$Q(s,a)$$ batch tuple|`(batch_size, num_discrete_actions)` x `num_discrete_branches`|