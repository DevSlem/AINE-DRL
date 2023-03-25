from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from aine_drl.policy.policy import PolicyDistParam


class Trainer:
    """
    PyTorch optimizer wrapper for single scalar loss.
    """
    @dataclass(frozen=True)
    class __ClipGradNormConfig:
        parameters: torch.Tensor | Iterable[torch.Tensor]
        max_norm: float
        norm_type: float = 2.0
        error_if_nonfinite: bool = False
    
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer
        self._clip_grad_norm_config = None
        
    def enable_grad_clip(
        self,
        parameters: torch.Tensor | Iterable[torch.Tensor],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False
    ) -> "Trainer":
        """
        Enable gradient clipping when parameter update. Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.

        Args:
            parameters (Tensor | Iterable[Tensor]): an iterable of Tensors or a single Tensor that will have gradients normalized
            max_norm (float): max norm of the gradients
            norm_type (float, optional): type of the used p-norm. Can be `inf` for infinity norm.
            error_if_nonfinite (bool, optional): if True, an error is thrown if the total norm of the gradients from `parameters` is `nan`, `inf`, or `-inf`. Default: False (will switch to True in the future)

        Returns:
            Trainer: self
        """
        self._clip_grad_norm_config = Trainer.__ClipGradNormConfig(
            parameters,
            max_norm,
            norm_type,
            error_if_nonfinite
        )
        return self
    
    def step(self, loss: torch.Tensor, training_steps: int):
        """
        Performs a single optimization step (parameter update).
        """
        self._optimizer.zero_grad()
        loss.backward()
        if self._clip_grad_norm_config is not None:
            clip_grad_norm_(**self._clip_grad_norm_config.__dict__)
        self._optimizer.step()

class NetworkTypeError(TypeError):
    def __init__(self, true_net_type: type) -> None:
        message = f"network must be inherited from \"{true_net_type.__name__}\"."
        super().__init__(message)
    
class Network(ABC):
    """
    AINE-DRL network abstract class.
    """
    @abstractmethod
    def model(self) -> nn.Module:
        raise NotImplementedError
    
    @property
    def device(self) -> torch.device:
        return next(self.model().parameters()).device
    
class RecurrentNetwork(Network):
    """
    Recurrent neural network (RNN) abstract class.
    """
    @property
    @abstractmethod
    def hidden_state_shape(self) -> tuple[int, int]:
        """
        Returns the shape of the rucurrent hidden state `(D x num_layers, H)`.
        
        * `num_layers`: the number of recurrent layers
        * `D`: 2 if bidirectional otherwise 1
        * `H`: the value depends on the type of the recurrent network

        When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
        """
        raise NotImplementedError
    
    @staticmethod
    def unpack_seq_shape(seq: torch.Tensor) -> tuple[int, int, torch.Size]:
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
    
class CategoricalLayer(nn.Module):
    """
    Linear layer for the discrete action type.

    Args:
        in_features (int): number of input features
        num_discrete_actions (int | tuple[int, ...]): each element indicates number of discrete actions of each branch
        is_logits (bool): whether logits or probabilities. Defaults to logits.
    """

    def __init__(self, in_features: int, 
                 num_discrete_actions: int | tuple[int, ...], 
                 is_logits: bool = True,
                 bias: bool = True,
                 device: torch.device | None = None,
                 dtype: Any | None = None) -> None:
        """
        Linear layer for the discrete action type.

        Args:
            in_features (int): number of input features
            num_discrete_actions (int | tuple[int, ...]): each element indicates number of discrete actions of each branch
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
    
class GaussianLayer(nn.Module):
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