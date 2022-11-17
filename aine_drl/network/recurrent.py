from ..network import Network
from ..policy.policy_distribution import PolicyDistributionParameter
from typing import Tuple, List, Union
import torch
from abc import abstractmethod, ABC
from dataclasses import dataclass

@dataclass(frozen=True)
class RecurrentHiddenState(ABC):
    """Recurrent hidden state wrapping abstract data class."""
    
    @property
    @abstractmethod
    def item(self) -> Tuple[torch.Tensor, ...]:
        """Unwrapping."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def shape(self) -> torch.Size:
        raise NotImplementedError
    
    @abstractmethod
    def __getitem__(self, idx) -> "RecurrentHiddenState":
        """batch indexing or slicing."""
        raise NotImplementedError
    
    @abstractmethod
    def __mul__(self, other: torch.Tensor) -> "RecurrentHiddenState":
        raise NotImplementedError
    
@dataclass(frozen=True)
class LSTMHiddenState(RecurrentHiddenState):
    """
    LSTM hidden state tuple wrapping data class. 
    Tensor shape is `(D x num_layers, batch_size, H_out)`. 
    See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html.

    Args:
        lstm_hidden_state (Tuple[Tensor, Tensor]): (h, c) whose shape is same
    """
    lstm_hidden_state: Tuple[torch.Tensor, torch.Tensor]
    
    @property
    def item(self) -> Tuple[torch.Tensor, ...]:
        return self.lstm_hidden_state
    
    @property
    def shape(self) -> torch.Size:
        return self.lstm_hidden_state[0].shape
    
    def __getitem__(self, idx) -> "RecurrentHiddenState":
        return LSTMHiddenState((self.lstm_hidden_state[0][:, idx], self.lstm_hidden_state[1][:, idx]))
    
    def __mul__(self, other: torch.Tensor) -> "RecurrentHiddenState":
        return LSTMHiddenState((self.lstm_hidden_state[0] * other, self.lstm_hidden_state[1] * other))
    
@dataclass(frozen=True)
class GRUHiddenState(RecurrentHiddenState):
    """
    GRU hidden state wrapping data class. 
    Tensor shape is `(D x num_layers, batch_size, H_out)`.
    See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
    
    Args:
        gru_hidden_state (Tensor): h
    """
    gru_hidden_state: torch.Tensor
    
    @property
    def item(self) -> Tuple[torch.Tensor, ...]:
        return (self.gru_hidden_state,)
    
    @property
    def shape(self) -> torch.Size:
        return self.gru_hidden_state.shape
    
    def __getitem__(self, idx) -> "RecurrentHiddenState":
        return GRUHiddenState(self.gru_hidden_state[:, idx])
    
    def __mul__(self, other: torch.Tensor) -> "RecurrentHiddenState":
        return GRUHiddenState(self.gru_hidden_state * other)
    
class RecurrentHiddenStateUtil(ABC):
    """Recurrent hidden state utility class."""
    @abstractmethod
    def create_inital_hidden_state(self) -> RecurrentHiddenState:
        raise NotImplementedError
    
    @abstractmethod
    def wrap(self, item: Tuple[torch.Tensor, ...]) -> RecurrentHiddenState:
        raise NotImplementedError
    
    @abstractmethod
    def to_batch(self, hidden_states: List[RecurrentHiddenState]) -> RecurrentHiddenState:
        raise NotImplementedError
    
class LSTMHiddenStateUtil(RecurrentHiddenStateUtil):
    def __init__(self, shape: Union[tuple, torch.Size]) -> None:
        self.shape = shape
        
    def create_inital_hidden_state(self) -> RecurrentHiddenState:
        return LSTMHiddenState((torch.zeros(self.shape), torch.zeros(self.shape)))

    def wrap(self, item: Tuple[torch.Tensor, ...]) -> RecurrentHiddenState:
        return LSTMHiddenState(item)
    
    def to_batch(self, hidden_states: List[RecurrentHiddenState]) -> RecurrentHiddenState:
        short_term_memories = []
        long_term_memories = []
        for hs in hidden_states:
            short_term_memories.append(hs.item[0])
            long_term_memories.append(hs.item[1])
            
        short_term_memories = torch.cat(short_term_memories, dim=1)
        long_term_memories = torch.cat(long_term_memories, dim=1)
        
        return LSTMHiddenState((short_term_memories, long_term_memories))
    
class GRUHiddenStateUtil(RecurrentHiddenStateUtil):
    
    def wrap(self, item: Tuple[torch.Tensor, ...]) -> RecurrentHiddenState:
        return GRUHiddenState(item[0])
    
    def to_batch(self, hidden_states: List[RecurrentHiddenState]) -> RecurrentHiddenState:
        long_short_term_memories = []
        for hs in hidden_states:
            long_short_term_memories.append(hs.item[0])
        
        long_short_term_memories = torch.cat(long_short_term_memories, dim=1)
        return GRUHiddenState(long_short_term_memories)

class RecurrentNetwork(Network):
    """
    Standard recurrent neural network (RNN).
    """
    
    @abstractmethod
    def create_hidden_state_wrapper(self) -> RecurrentHiddenStateUtil:
        raise NotImplementedError
    
class RecurrentActorCriticNetwork(RecurrentNetwork):
    # 1. set the initial state of an episode to zero.
    # 2. how can you track initial states???
    # 3. when training recurrent PPO, you can consider two options. either random sampling training or recursive training, what is the best?
    # 4. isn't random sampling training violated for RNN mechanism, especially backpropagation??
    # 5. when sequence length is greater than 1, how to do??

    # Unity ML-Agents의 경우 Recurrent PPO에 random sampling 방식을 취하는 중
    # sequence length가 1보다 클 경우 input을 stacking할 필요가 있는 듯?
    
    @abstractmethod
    def forward(self, 
                obs: torch.Tensor, 
                hidden_state: RecurrentHiddenState) -> Tuple[PolicyDistributionParameter, torch.Tensor, RecurrentHiddenState]:
        """
        Compute policy distribution parameters whose shape is `(batch_size, ...)`, 
        state value whose shape is `(batch_size, 1)` and next recurrent hidden state. \\
        `batch_size` is `num_envs` x `n-step`. \\
        When the action type is discrete, policy distribution is generally logits or soft-max distribution. \\
        When the action type is continuous, it's generally mean and standard deviation of gaussian distribution. \\
        Recurrent hidden state is typically `LSTMHiddenState` or `GRUHiddenState`.

        Args:
            obs (Tensor): observation of state whose shape is `(batch_size, *obs_shape)`

        Returns:
            Tuple[PolicyDistributionParameter, Tensor, RecurrentHiddenState]: policy distribution parameter, state value, recurrent hidden state
        """
        pass
