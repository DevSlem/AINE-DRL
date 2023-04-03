---
sort: 7
---

# Env

AINE-DRL compatible reinforcement learning environment interface.

**Module**: `aine_drl.env`

```python
class Env(ABC)
```

## Properties

### num_envs

Returns the number of environments.

```python
@property
@abstractmethod
def num_envs(self) -> int
```

### obs_spaces

Returns the shapes of the observation spaces.

```python
@property
@abstractmethod
def obs_spaces(self) -> tuple[ObservationSpace, ...]
```

### action_space

Returns action space of the environment.

```python
@property
@abstractmethod
def action_space(self) -> ActionSpace
```

## Methods

### reset()

Resets the environment to an initial state and returns the initial observation.

```python
@abstractmethod
def reset(self) -> Observation
```

Returns:

|Name|Description|Shape|
|---|---|---|
|obs (`Observation`)|Observation of the initial state.|`*batch_shape` = `(num_envs,)` details in `Observation` docs|

### step()

Takes a step in the environment using an action.

```python
@abstractmethod
def step(
    self, 
    action: Action
) -> tuple[Observation, torch.Tensor, torch.Tensor, Observation | None]
```

Parameters:

|Name|Description|Shape|
|---|---|---|
|action (`Action`)|Action provided by the agent.|`*batch_shape` = `(num_envs,)` details in `Action` docs|

Returns:

|Name|Description|Shape|
|---|---|---|
|next_obs (`Observation`)|Next observation which is automatically reset to the first observation of the next episode.|`*batch_shape` = `(num_envs,)` details in `Observation` docs|
|reward (`Tensor`)|Scalar reward values|`(num_envs, 1)`|
|terminated (`Tensor`)|Whether the episode is terminated|`(num_envs, 1)`|
|real_final_next_obs (`Observation | None`)|"Real" final next observation of the episode. You can access only if any environment is terminated.|`*batch_shape` = `(num_terminated_envs,)` details in `Observation` docs|

### close()

Close all environments and release resources.

```python
@abstractmethod
def close(self)
```