---
sort: 1
---


# REINFORCE

**REINFORCE** is a simple and basic policy gradient method based on Monte Carlo (MC) method. Differnt from value-based method, it has a simple idea: policy is a parameterized function itself!

Paper: [Policy Gradient Methods for Reinforcement Learning with Function Approximation ](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

REINFORCE has below features:

* policy-based
* Monte Carlo (MC) method (episodic)
* on-policy
* no bias, high variance

Since REINFORCE is MC method, it computes return \$$G_t$$ which is used to update policy parameters. It must wait to update parameters until an episode is terminated to compute return \$$G_t$$. 

REINFORCE high variance can be reduced using baseline \$$b(s)$$. It's called **REINFORCE with Baseline**. REINFORCE agent uses \$$G_t - b(s)$$ instead of just \$$G_t$$, and baseline \$$b(s)$$ is the mean of returns.

You can see source code in [reinforce](https://github.com/DevSlem/AINE-DRL/tree/main/aine_drl/agent/reinforce).

## Configuration

Since it has simple hyperparameters, you don't need to understand deeply reinforcement learning. 

> Note if the setting has default value, you can skip it.

|Setting|Description|
|---|---|
|`gamma`|(`float`, default = `0.99`) Discount factor \$$\gamma$$ of future rewards.|
|`entropy_coef`|(`float`, default = `0.001`) Entropy multiplier used to compute loss. It adjusts exploration/exploitation balance.|

## Network

class: `REINFORCENetwork`

You need to implement below methods.

### Forward

```python
@abstractmethod
def forward(
    self, 
    obs: Observation
) -> PolicyDist
```

|Input|Description|Shape|
|---|---|---|
|obs (`Observation`)|observation batch tuple|`*batch_shape` = `(batch_size,)` details in `Observation` docs|

|Output|Description|Shape|
|---|---|---|
|policy_dist (`PolicyDist`)|policy distribution \$$\pi(a \vert s)$$|`*batch_shape` = `(batch_size,)` details in `PolicyDist` docs|