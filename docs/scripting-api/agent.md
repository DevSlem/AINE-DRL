---
sort: 1
---

# Agent

An agent is an actor that can observe and interact with the environment. Depending on the reinforcement learning algorithms, the way the agent is trained and selects actions is different.

If you want to implement your own agent, you can inherit from the abstract class `Agent` and implement the abstract methods.

**Module**: `aine_drl.agent`

```python
class Agent(ABC)
```

## Constructor

```python
def __init__(
    self,
    num_envs: int,
    network: Network,
    behavior_type: BehaviorType = BehaviorType.TRAIN,
)
```

Parameters:

|Name|Description|
|---|---|
|num_envs (`int`)|The number of environments the agent is interacting with.|
|network (`Network`)|The network of the agent.|
|behavior_type (`BehaviorType`)|(default = `BehaviorType.TRAIN`) The behavior type of the agent.|

## Properties

### name

The name of the agent. You need to implement this property when you inherit from `Agent`.

```python
@property
@abstractmethod
def name(self) -> str
```

### device

The device where the agent is running.

```python
@property
def device(self) -> torch.device
```

### num_envs

The number of environments the agent is interacting with. This is the number of environments in the vectorized environment.

```python
@property
def num_envs(self) -> int
```

### training_steps

The number of training steps the network has been trained for. This equals to the times the optimizer take steps.

```python
@property
def training_steps(self) -> int
```

### behavior_type

The behavior type of the agent. When you train the agent, the behavior type is `BehaviorType.TRAIN`. When you inference the agent, the behavior type is `BehaviorType.INFERENCE`.

```python
@property
def behavior_type(self) -> BehaviorType
```

```python
@behavior_type.setter
def behavior_type(self, value: BehaviorType)
```

### log_keys

The keys of log data that the agent want to log. You need to override this property when you want to log some data.

```python
@property
def log_keys(self) -> tuple[str, ...]
```

### log_data

The log data of the agent. The data is a dictionary with the keys in `log_keys` and values with tuple `(value, time)`. You need to override this property when you want to log some data.

```python
@property
def log_data(self) -> dict[str, tuple[Any, float]]
```

### state_dict

The state dictionary of the agent. The state dictionary contains the state of the agent, including the network and other data. You can save them to a PyTorch file and load them later. You need to override this property when you want to save the state of the agent.

```python
@property
def state_dict(self) -> dict
```

## Methods

### select_action()

Select actions from the `Observation`.

```python
def select_action(self, obs: Observation) -> Action
```

Parameters:

|Name|Description|Shape|
|---|---|---|
|obs (`Observation`)|One-step observation batch tuple.|`*batch_shape` = `(num_envs,)` details in `Observation` docs|

Returns:

|Name|Description|Shape|
|---|---|---|
|action (`Action`)|One-step action batch.|`*batch_shape` = `(num_envs,)` details in `Action` docs|

### _select_action_train()

Select actions from the `Observation` when training. You need to implement this method when you inherit from `Agent`.

```python
@abstractmethod
def _select_action_train(self, obs: Observation) -> Action
```

The parameters and return values are the same as `select_action()`.

### _select_action_inference()

Select actions from the `Observation` when inference. You need to implement this method when you inherit from `Agent`.

```python
@abstractmethod
def _select_action_inference(self, obs: Observation) -> Action
```

The parameters and return values are the same as `select_action()`.

### update()

Update and train the agent.

```python
def update(self, exp: Experience)
```

Parameters:

|Name|Description|Shape|
|---|---|---|
|exp (`Experience`)|One-step experience tuple.|`*batch_shape` = `(num_envs,)` details in `Experience` docs|

### _update_train()

Update and train the agent when training. You need to implement this method when you inherit from `Agent`.

```python
@abstractmethod
def _update_train(self, exp: Experience)
```

The parameters are the same as `update()`.

### _update_inference()

Update and train the agent when inference. You need to implement this method when you inherit from `Agent`.

```python
@abstractmethod
def _update_inference(self, exp: Experience)
```

The parameters are the same as `update()`.

### _tick_training_steps()

Tick the training steps.

```python
def _tick_training_steps(self)
```

### load_state_dict()

Load the state dictionary of the agent. You need to override this method when you want to load the state of the agent.

```python
def load_state_dict(self, state_dict: dict)
```