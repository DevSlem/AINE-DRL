---
sort: 5
---

# Action

Action data type with tensor.

`*batch_shape` depends on the input of the algorithm you are using.

* simple batch: `*batch_shape` = `(batch_size,)`
* sequence batch: `*batch_shape` = `(seq_batch_size, seq_len)`

`discrete_action` and `continuous_action` are tensors of the discrete and continuous action spaces respectively. If the action space is only discrete, `continuous_action` is empty tensor `(*batch_shape, 0)` and the other case is vice versa.

**Module**: `aine_drl.exp`

```python
@dataclass(frozen=True)
class Action
```

## Fields

### discrete_action

The discrete action tensor `(*batch_shape, num_discrete_branches)`.

```python
discrete_action: torch.Tensor
```

### continuous_action

The continuous action tensor `(*batch_shape, num_continuous_branches)`.

```python
continuous_action: torch.Tensor
```

## Properties

### num_discrete_branches

The number of discrete action branches.

```python
@property
def num_discrete_branches(self) -> int
```

### num_continuous_branches

The number of continuous action branches.

```python
@property
def num_continuous_branches(self) -> int
```

### num_branches

The number of action branches which is equal to `num_discrete_branches` + `num_continuous_branches`.

```python
@property
def num_branches(self) -> int
```

### batch_shape

The batch shape `*batch_shape` of the action tensor.

```python
@property
def batch_shape(self) -> torch.Size
```

## Methods

### transform()

Transform the action tensor with the callable function.

```python
def transform(self, func: Callable[[Tensor], Tensor]) -> Action
```

Parameters:

|Name|Description|
|---|---|
|func (`(Tensor) -> Tensor`)|The callable function to transform each action tensor. |

Returns:

|Name|Description|
|---|---|
|action (`Action`)|The transformed action.|

### __getitem__()

Get a batch of `Action` from the `Action` instance. Note that it's recommended to use range slicing instead of indexing.

```python
def __getitem__(self, idx) -> Action
```

### from_iter()

Create an `Action` batch instance from iterable of `Action`. Each item in the iterable must consist of the same action spaces. For example, if first `Action` has only discrete actions and the number of discrete action branches is `4`, second `Action` must be same.

```python
@staticmethod
def from_iter(actions: Iterable[Action]) -> Action
```

Example:

```python
import torch
from aine_drl.exp import Action

batch_size1 = 2
batch_size2 = 3

action1 = Action(
    continuous_action=torch.randn(batch_size1, 2)
)

action2 = Action(
    continuous_action=torch.randn(batch_size2, 2)
)

action = Action.from_iter([action1, action2])
>>> action.discrete_action.shape, action.continuous_action.shape
(torch.Size([5, 0]), torch.Size([5, 2]))
```