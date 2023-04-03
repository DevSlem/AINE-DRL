---
sort: 2
---

# Network

A network is a neural network that is used by an agent to predict the action given the observation. You can implement your own network by inheriting from the abstract class `Network` and implementing the abstract methods.

**Module**: `aine_drl.net`

```python
class Network(ABC)
```

## Properties

### device

The device where the network is running.

```python
@property
def device(self) -> torch.device
```

### Methods

### model()

The model of the network. The `model()` must be return `torch.nn.Module` object which includes all the layers of the network. 

```python
@abstractmethod
def model(self) -> torch.nn.Module
```

For example, assume that your network consists of encoding layer, actor layer, critic layer. Then, the `model()` method should return the following `torch.nn.Module` object:

```python
import torch.nn as nn

class FooNet(nn.Module, Network):
    def __init__(self):
        super().__init__()

        self._encoder = nn.Linear(4, 64)
        self._actor = nn.Linear(64, 2)
        self._critic = nn.Linear(64, 1)
    
    def model(self):
        return self
```
