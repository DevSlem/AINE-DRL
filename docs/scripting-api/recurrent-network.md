---
sort: 3
---

# RecurrentNetwork

A recurrent network is a neural network that is used by an agent to predict the action given the observation. You can implement your own recurrent network by inheriting from the abstract class `RecurrentNetwork` and implementing the abstract methods.

**Module**: `aine_drl.net`

```python
class RecurrentNetwork(Network)
```

It's inherited from `Network`. See [Network](./network.md) docs.

## Methods

### hidden_state_shape()

Returns the shape of the rucurrent hidden state `(D x num_layers, H)`.

* `num_layers`: the number of recurrent layers
* `D`: 2 if bidirectional otherwise 1
* `H`: the value depends on the type of the recurrent network

When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. <br>
When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.

```python
@abstractmethod
def hidden_state_shape(self) -> tuple[int, int]
```
