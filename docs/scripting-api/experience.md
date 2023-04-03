---
sort: 6
---

# Experience

Experience tuple at time step $$t$$. It contains the observation, action, next observation, reward, terminated. Each item shape is `(num_envs, *item_shape)`. `*item_shape` depends on the item. 

## Fields

### obs

The observation tuple. See [Observation](./observation.md) docs.

```python
obs: Observation
```

### action

See [Action](./action.md) docs.

```python
action: Action
```

### next_obs

The next observation tuple. See [Observation](./observation.md) docs.

```python
next_obs: Observation
```

### reward

The reward tensor. `*item_shape` = `1`

```python
reward: Tensor
```

### terminated

The terminated tensor. `*item_shape` = `1`

```python
terminated: Tensor
```

## Methods

### transform()

Transform the experience tuple with the callable function.

```python
def transform(self, func: Callable[[Tensor], Tensor]) -> Experience
```

Parameters:

|Name|Description|
|---|---|
|func (`(Tensor) -> Tensor`)|The callable function to transform each experience tensor.|

Returns:

|Name|Description|
|---|---|
|exp (`Experience`)|The transformed experience.|