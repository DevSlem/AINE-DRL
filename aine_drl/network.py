from abc import ABC, abstractmethod
from typing import Tuple, Union, Any, Generic, TypeVar, Dict, Iterator
from aine_drl.policy.policy_distribution import PolicyDistParam
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

T = TypeVar("T")

class NetworkTypeError(TypeError):
    def __init__(self, true_net_type: type) -> None:
        message = f"The network must be inherited from \"{true_net_type.__name__}\"."
        super().__init__(message)

class DiscreteActionLayer(nn.Module):
    """
    Linear layer for the discrete action type.

    Args:
        in_features (int): number of input features
        num_discrete_actions (int | Tuple[int, ...]): each element indicates number of discrete actions of each branch
        is_logits (bool): whether logits or probabilities. Defaults to logits.
    """

    def __init__(self, in_features: int, 
                 num_discrete_actions: Union[int, Tuple[int, ...]], 
                 is_logits: bool = True,
                 bias: bool = True,
                 device: torch.device | None = None,
                 dtype: Any | None = None) -> None:
        """
        Linear layer for the discrete action type.

        Args:
            in_features (int): number of input features
            num_discrete_actions (int | Tuple[int, ...]): each element indicates number of discrete actions of each branch
            is_logits (bool): whether logits or probabilities. Defaults to logits.
        """
        super().__init__()
        
        self.is_logits = is_logits
        
        if type(num_discrete_actions) is int:
            num_discrete_actions = (num_discrete_actions,)
        self.num_discrete_actions = num_discrete_actions
        
        self.total_num_discrete_actions = 0
        for num_action in num_discrete_actions: # type: ignore
            self.total_num_discrete_actions += num_action
        
        self.layer = nn.Linear(
            in_features,
            self.total_num_discrete_actions,
            bias,
            device,
            dtype
        )
    
    def forward(self, x: torch.Tensor) -> PolicyDistParam:
        out = self.layer(x)
        discrete_pdparams = torch.split(out, self.num_discrete_actions, dim=1)
        
        if not self.is_logits:
            discrete_pdparams = list(discrete_pdparams)
            for i in range(len(discrete_pdparams)):
                discrete_pdparams[i] = F.softmax(discrete_pdparams[i], dim=1)
            discrete_pdparams = tuple(discrete_pdparams)
        
        return PolicyDistParam(discrete_pdparams=discrete_pdparams)
    
class GaussianContinuousActionLayer(nn.Module):
    """
    Linear layer for the continuous action type.

    Args:
        in_features (int): number of input features
        num_continuous_actions (int): number of continuous actions which equals to `num_continuous_branches`
    """
    
    def __init__(self, in_features: int, 
                 num_continuous_actions: int, 
                 is_log_std: bool = True,
                 bias: bool = True,
                 device: torch.device | None = None,
                 dtype: Any | None = None) -> None:
        """
        Linear layer for the continuous action type.

        Args:
            in_features (int): number of input features
            num_continuous_actions (int): number of continuous actions which equals to `num_continuous_branches`
        """
        super().__init__()
        
        self.num_continuous_actions = num_continuous_actions
        self.is_log_std = is_log_std
        
        self.layer = nn.Linear(
            in_features,
            self.num_continuous_actions * 2,
            bias,
            device,
            dtype
        )
        
    def forward(self, x: torch.Tensor) -> PolicyDistParam:
        out = self.layer(x)
        return PolicyDistParam(continuous_pdparams=torch.split(out, 2, dim=1))
    
class Network(ABC, Generic[T]):
    """
    AINE-DRL network abstract class.
    """
    
    def __init__(self) -> None:
        self._models: Dict[str, nn.Module] = {}
        self._device = torch.device("cpu")
    
    @abstractmethod
    def train_step(self, 
                   loss: T,
                   training_step: int):
        """
        Gradient step for training.

        Args:
            loss (T): computed loss, `T` is generic type
            training_step (int): current training step
        """
        raise NotImplementedError
    
    @property
    def device(self) -> torch.device:
        """
        Returns:
            torch.device: device of the network
        """
        return self._device
    
    def to(self, device: torch.device | None = None) -> "Network[T]":
        """
        Move the network to the device.

        Args:
            device (torch.device | None): device to move. If `None`, move to CPU. Defaults to `None`.
        """
        if device is None:
            device = torch.device("cpu")
        self._device = device
        for model in self._models.values():
            model.to(device)
        return self
        
    def add_model(self, name: str, model: nn.Module):
        """
        Add the model to the network. Note that you must add at least one model.
        """
        self._models[name] = model
    
    def state_dict(self) -> dict:
        """Returns all model state dict."""
        return {name: model.state_dict() for name, model in self._models.items()}
    
    def load_state_dict(self, state_dict: dict):
        """Loads all model state dict."""
        for name, model_state_dict in state_dict.items():
            self._models[name].load_state_dict(model_state_dict)
            
    def parameters(self) -> Iterator[Parameter]:
        """
        Returns all model parameters.
        """
        return self.concat_model_params(*self._models.values())
        
    @staticmethod
    def model_device(model: nn.Module) -> torch.device:
        """Returns the device of the model."""
        return next(model.parameters()).device
    
    @staticmethod
    def concat_model_params(*models: nn.Module) -> Iterator[Parameter]:
        """Concatenate model parameters."""
        params = []
        for model in models:
            params.extend(list(model.parameters()))
        return params  # type: ignore

class RecurrentNetwork(Network[T]):
    """
    Recurrent neural network (RNN) abstract class.
    """
    @property
    @abstractmethod
    def hidden_state_shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the rucurrent hidden state `(D x num_layers, H)`. \\
        `num_layers` is the number of recurrent layers. \\
        `D` = 2 if bidirectional otherwise 1. \\
        The value of `H` depends on the type of the recurrent network.
        When you use LSTM, `H` = `H_out x 2`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
        """
        raise NotImplementedError
    
    @staticmethod
    def unpack_seq_shape(seq: torch.Tensor) -> Tuple[int, int, torch.Size]:
        """
        Unpack the sequence shape.
        
        Args:
            seq (Tensor): `(num_seq, seq_len, *feature_shape)`
        
        Returns:
            num_seq (int): the number of sequences
            seq_len (int): the length of each sequence
            feature_shape (torch.Size): the shape of each feature (excluding the sequence dimension
        """
        seq_shape = seq.shape
        return seq_shape[0], seq_shape[1], seq_shape[2:]
    
    @staticmethod
    def pack_lstm_hidden_state(lstm_hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """`(D x num_layers, num_seq, H_out) x 2` -> `(D x num_layers, num_seq, H_out x 2)`"""
        return torch.cat(lstm_hidden_state, dim=2)
    
    @staticmethod
    def unpack_lstm_hidden_state(lstm_hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(D x num_layers, num_seq, H_out x 2)` -> `(D x num_layers, num_seq, H_out) x 2`"""
        lstm_hidden_state = lstm_hidden_state.split(lstm_hidden_state.shape[2] // 2, dim=2)  # type: ignore
        return (lstm_hidden_state[0].contiguous(), lstm_hidden_state[1].contiguous())
    