---
sort: 4
---

# Observation

Observation data type with tensor tuple. It can have multiple observation spaces.

Each item shape is `(*batch_shape, *obs_shape)`. `*obs_shape` depends on the observation space. Example:

* vector: `*obs_shape` = `(features,)`
* image: `*obs_shape` = `(height, width, channel)`

`*batch_shape` depends on the input of the algorithm you are using:

* simple batch: `*batch_shape` = `(batch_size,)`
* sequence batch: `*batch_shape` = `(seq_batch_size, seq_len)`

You can subscript the `Observation` instance to get a batch of `Observation`.
Note that it works to all observation `Tensor` instances.

**Module**: `aine_drl.exp`
    
```python
@dataclass(frozen=True)
class Observation
```

## Fields

### items

The observation tensor tuple. Each item is different depending on the observation space. For example, `items[0]` is vector and `items[1]` is image.

```python
items: tuple[Tensor, ...]
```

## Properties

### num_items

The number of items in the observation.

```python
@property
def num_items(self) -> int
```

## Methods

### transform()

Transform the observation tensor tuple with the callable function.

```python
def transform(self, func: Callable[[Tensor], Tensor]) -> Observation
```

Parameters:

|Name|Description|
|---|---|
|func (`(Tensor) -> Tensor`)|The callable function to transform each observation tensor.|

Returns:

|Name|Description|
|---|---|
|obs (`Observation`)|The transformed observation.|

### __getitem__()

Get a batch of `Observation` from the `Observation` instance. Note that it's recommended to use range slicing instead of indexing.

```python
def __getitem__(self, idx) -> Observation
```

### __setitem__()

Set a batch of `Observation` to the `Observation` instance. Note that it's recommended to use range slicing instead of indexing.

```python
def __setitem__(self, idx, value: Observation)
```

### from_tensor()

Create an `Observation` instance from tensors.

```python
@staticmethod
def from_tensor(*obs_tensors: Tensor) -> Observation
```

### from_iter()

Create an `Observation` batch instance from iterable of `Observation`. Each item in the iterable must consist of the same observation spaces. For example, if first `Observation` has vector and image, second `Observation` must have vector and image with same dimensions and so on.

```python
@staticmethod
def from_iter(iter: Iterable[Observation]) -> Observation
```

Example:

```python
import torch
from aine_drl.exp import Observation

batch_size1 = 2
batch_size2 = 3

obs1 = Observation.from_tensor(
    torch.randn(batch_size1, 3), 
    torch.randn(batch_size1, 4, 5, 3)
)
obs2 = Observation.from_tensor(
    torch.randn(batch_size2, 3), 
    torch.randn(batch_size2, 4, 5, 3)
)

obs = Observation.from_iter([obs1, obs2])
>>> obs.items[0].shape, obs.items[1].shape
(torch.Size([5, 3]), torch.Size([5, 4, 5, 3]))
```