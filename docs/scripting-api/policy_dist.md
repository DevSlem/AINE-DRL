---
sort: 8
---

# PolicyDist

Policy distribution interface.

`*batch_shape` depends on the input of the algorithm you are using.

* simple batch: `*batch_shape` = `(batch_size,)`
* sequence batch: `*batch_shape` = `(seq_batch_size, seq_len)`

**Module**: `aine_drl.policy_dist`

```python
class PolicyDist(ABC)
```

## Methods

### sample()

Sample actions from the policy distribution.

```python
@abstractmethod
def sample(self, reparam_trick: bool = False) -> Action
```

Parameters:

|Name|Description|
|---|---|
|reparam_trick (`bool`)|(default = `False`) Whether to use reparameterization trick.|

Returns:

|Name|Shape|
|---|---|
|action (`Action`)|Action shape depends on the constructor arguments|

### log_prob()

Returns the log of the probability mass/density function according to the `Action`.

```python
@abstractmethod
def log_prob(self, action: Action) -> torch.Tensor
```

Returns:

|Name|Shape|
|---|---|
|log_prob (`Tensor`)|`(*batch_shape, num_branches)`|

### joint_log_prob()

Returns the joint log of the probability mass/density function according to the `action`.

```python
def joint_log_prob(self, action: Action) -> torch.Tensor
```

Returns:

|Name|Shape|
|---|---|
|joint_log_prob (`Tensor`)|`(*batch_shape, 1)`|

### entropy()

Returns the entropy of the policy distribution.

```python
@abstractmethod
def entropy(self) -> torch.Tensor
```

Returns:

|Name|Shape|
|---|---|
|entropy (`Tensor`)|`(*batch_shape, num_branches)`|

### joint_entropy()

Returns the joint entropy of the policy distribution.

```python
def joint_entropy(self) -> torch.Tensor
```

Returns:

|Name|Shape|
|---|---|
|joint_entropy (`Tensor`)|`(*batch_shape, 1)`|