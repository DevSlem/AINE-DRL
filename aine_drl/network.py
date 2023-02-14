from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Any, Generic, TypeVar, Dict
from aine_drl.policy.policy_distribution import PolicyDistParam
import torch
import torch.nn as nn
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
                 device: Optional[torch.device] = None,
                 dtype: Optional[Any] = None) -> None:
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
        for num_action in num_discrete_actions:
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
                 device: Optional[torch.device] = None,
                 dtype: Optional[Any] = None) -> None:
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
                   grad_clip_max_norm: Optional[float],
                   training_step: int):
        """
        Gradient step for training.

        Args:
            loss (T): computed loss, `T` is generic type
            grad_clip_max_norm (float | None): maximum norm for the gradient clipping
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
    
    def to(self, device: Optional[torch.device] = None) -> "Network[T]":
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
            
    def parameters(self) -> List[nn.parameter.Parameter]:
        """
        Returns all model parameters.
        """
        return self.concat_model_params(*self._models.values())
          
    @staticmethod
    def simple_train_step(loss: torch.Tensor,
                          optimizer: torch.optim.Optimizer,
                          grad_clip_max_norm: Optional[float] = None):
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], grad_clip_max_norm)
        optimizer.step()
        
    @staticmethod
    def model_device(model: nn.Module) -> torch.device:
        """Returns the device of the model."""
        return next(model.parameters()).device
    
    @staticmethod
    def concat_model_params(*models: nn.Module) -> List[nn.parameter.Parameter]:
        """Concatenate model parameters."""
        params = []
        for model in models:
            params.extend(list(model.parameters()))
        return params

class VNetwork(Network):
    """
    State Value Function V(s) Estimator.
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value V. \\
        `batch_size` equals to `num_envs x n_steps`.

        Args:
            obs (Tensor): observation of state whose shape is `(batch_size, *obs_shape)`

        Returns:
            Tensor: state value whose shape is `(batch_size, 1)`
        """
        raise NotImplementedError
    
class QNetwork(Network):
    """
    Action Value Function Q(s,a) Estimator. It's generally used for continuous action type.
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Estimate action value Q. \\
        `batch_size` equals to `num_envs x n_steps`. \\
        Observation (or feature extracted from it) should be concatenated with action and then feed forward it.
        
        Args:
            obs (Tensor): observation of state whose shape is `(batch_size, *obs_shape)`
            action (Tensor): continuous action whose shape is `(batch_size, num_action_branches)`
            
        Returns:
            Tensor: action value whose shape is `(batch_size, 1)`
        """
        raise NotImplementedError

class ActorCriticSharedNetwork(Network):
    """
    Actor critic network.
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> Tuple[PolicyDistParam, torch.Tensor]:
        """
        Compute policy distribution parameters whose shape is `(batch_size, ...)`, 
        state value whose shape is `(batch_size, 1)`. \\
        `batch_size` is `num_envs` x `n-step`. \\
        When the action type is discrete, policy distribution is generally logits or soft-max distribution. \\
        When the action type is continuous, it's generally mean and standard deviation of gaussian distribution.

        Args:
            obs (Tensor): observation of state whose shape is `(batch_size, *obs_shape)`

        Returns:
            Tuple[PolicyDistParam, Tensor]: policy distribution parameter, state value
        """
        raise NotImplementedError

class QValueNetwork(Network):
    """
    Action value Q network.
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> PolicyDistParam:
        """
        Compute action value Q.  \\
        Note that it only works to discrete action type. 
        So, you must set only `PolicyDistParam.discrete_pdparams` which is action values. \\
        `batch_size` is `num_envs` x `n-step`.
        
        Args:
            obs (Tensor): observation of state whose shape is `(batch_size, *obs_shape)`
            
        Returns:
            PolicyDistParam: discrete action value
        """
        raise NotImplementedError

class RecurrentNetwork(Network):
    """
    Standard recurrent neural network (RNN).
    """
    @staticmethod
    def pack_lstm_hidden_state(lstm_hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """`(D x num_layers, num_seq, H_out) x 2` -> `(D x num_layers, num_seq, H_out x 2)`"""
        return torch.cat(lstm_hidden_state, dim=2)
    
    @staticmethod
    def unpack_lstm_hidden_state(lstm_hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(D x num_layers, num_seq, H_out x 2)` -> `(D x num_layers, num_seq, H_out) x 2`"""
        lstm_hidden_state = lstm_hidden_state.split(lstm_hidden_state.shape[2] // 2, dim=2)  # type: ignore
        return (lstm_hidden_state[0].contiguous(), lstm_hidden_state[1].contiguous())
    
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
    
class RecurrentActorCriticSharedNetwork(RecurrentNetwork):
    """Recurrent actor critic shared network."""
        
    @abstractmethod
    def forward(self, 
                obs_seq: torch.Tensor, 
                hidden_state: torch.Tensor) -> Tuple[PolicyDistParam, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution parameter (pdparam), state value, using the recurrent layer
        
        When the action type is discrete, pdparam is generally logits or soft-max distribution. \\
        When the action type is continuous, it's generally mean and standard deviation of gaussian distribution.
        
        It's recommended to set recurrent layer to `batch_first=True`.

        Args:
            obs_seq (Tensor): observation sequences
            hidden_state (Tensor): hidden states at the beginning of each sequence

        Returns:
            pdparam_seq (PolicyDistParam): policy distribution parameter sequences
            state_value_seq (Tensor): state value sequences
            next_hidden_state (Tensor): next hidden state
            
        ## Input/Output Details
        
        `num_layers` is the number of recurrent layers. \\
        `D` = 2 if bidirectional otherwise 1. \\
        The value of `H` depends on the type of the recurrent network.
        When you use LSTM, `H` = `H_out x 2`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs_seq|`(num_seq, seq_len, *obs_shape)`|
        |hidden state|`(D x num_layers, num_seq, H)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |pdparam_seq|`*batch_shape` = `(num_seq, seq_len)`, details in `PolicyDistParam` docs|
        |state_value_seq|`(num_seq, seq_len, 1)`|
        |next_hidden_state|`(D x num_layers, num_seq, H)`|
            
        ## Examples
        
        `forward()` method example when using LSTM::
        
            def forward(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[aine_drl.PolicyDistParam, torch.Tensor, torch.Tensor]:
                # feed forward to the encoding layer
                # (num_seq, seq_len, *obs_shape) -> (num_seq * seq_len, *obs_shape)
                seq_len = obs_seq.shape[1]
                obs_batch = obs_seq.flatten(0, 1)
                encoded_batch = self.encoding_layer(obs_batch)
                
                # LSTM layer
                unpacked_hidden_state = aine_drl.RecurrentNetwork.unpack_lstm_hidden_state(hidden_state)
                # (num_seq * seq_len, lstm_in_features) -> (num_seq, seq_len, lstm_in_features)
                encoded_seq = encoded_batch.reshape(-1, seq_len, self.lstm_in_features)
                encoded_seq, unpacked_next_hidden_state = self.lstm_layer(encoded_seq, unpacked_hidden_state)
                next_hidden_state = aine_drl.RecurrentNetwork.pack_lstm_hidden_state(unpacked_next_hidden_state)
                
                # actor-critic layer
                # (num_seq, seq_len, D x H_out) -> (num_seq * seq_len, D x H_out)
                encoded_batch = encoded_seq.flatten(0, 1)
                pdparam_batch = self.actor_layer(encoded_batch)
                state_value_batch = self.critic_layer(encoded_batch)
                
                # (num_seq * seq_len, *shape) -> (num_seq, seq_len, *shape)
                pdparam_seq = pdparam_batch.flattened_to_sequence(seq_len)
                state_value_seq = state_value_batch.reshape(-1, seq_len, 1)
                
                return pdparam_seq, state_value_seq, next_hidden_state
        """
        raise NotImplementedError
    
class RecurrentActorCriticSharedTwoValueNetwork(nn.Module):
    """
    Recurrent Actor Critic Shared Two Value Network. 
    It combines extrinsic and intrinsic reward streams. 
    Each stream can be different episodic or non-episodic, and can have different discount factors.  
    It constitutes of encoding layers with recurrent layer and of 3 output layers which are policy, extrinsic value and intrinsic value layers. 
    LSTM or GRU are commonly used as the recurrent layer.
    """
        
    @abstractmethod
    def forward(self, 
                obs_seq: torch.Tensor, 
                hidden_state: torch.Tensor) -> Tuple[PolicyDistParam, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution parameter (pdparam), extinrisc state value, intrinsic state value using the recurrent layer.
        
        When the action type is discrete, pdparam is generally logits or soft-max distribution. \\
        When the action type is continuous, it's generally mean and standard deviation of gaussian distribution.
        
        It's recommended to set recurrent layer to `batch_first=True`.
        
        Args:
            obs_seq (Tensor): observation sequences
            hidden_state (Tensor): hidden states at the beginning of each sequence

        Returns:
            pdparam_seq (PolicyDistParam): policy distribution parameter sequences
            ext_state_value_seq (Tensor): extrinsic state value sequences
            int_state_value_seq (Tensor): intrinsic state value sequences
            next_hidden_state (Tensor): next hidden state
        
        ## Input/Output Details
        
        `batch_size` is flattened sequence batch whose shape is `sequence_batch_size` x `sequence_length`.
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs_seq|`(num_seq, seq_len, *obs_shape)`|
        |hidden state|`(D x num_layers, num_seq, H)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |pdparam_seq|`*batch_shape` = `(num_seq, seq_len)`, details in `PolicyDistParam` docs|
        |ext_state_value_seq|`(num_seq, seq_len, 1)`|
        |int_state_value_seq|`(num_seq, seq_len, 1)`|
        |next_hidden_state|`(D x num_layers, num_seq, H)`|
        
        It's recommended to set the recurrent layer to `batch_first=True`. \\
        Hidden state is typically concatnated tensor of LSTM tuple (h, c) or GRU hidden state tensor.
            
        ## Examples
        
        `forward()` method example when using LSTM::
        
            def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[aine_drl.PolicyDistParam, torch.Tensor, torch.Tensor]:
                # encoding layer
                # (batch_size, seq_len, *obs_shape) -> (batch_size * seq_len, *obs_shape)
                seq_len = obs.shape[1]
                flattend = obs.flatten(0, 1)
                encoding = self.encoding_layer(flattend)
                
                # lstm layer
                unpacked_hidden_state = aine_drl.RecurrentNetwork.unpack_lstm_hidden_state(hidden_state)
                # (batch_size * seq_len, *lstm_in_feature) -> (batch_size, seq_len, *lstm_in_feature)
                encoding = encoding.reshape(-1, seq_len, self.lstm_in_feature)
                encoding, unpacked_hidden_state = self.lstm_layer(encoding, unpacked_hidden_state)
                next_hidden_state = aine_drl.RecurrentNetwork.pack_lstm_hidden_state(unpacked_hidden_state)
                
                # actor-critic layer
                # (batch_size, seq_len, *hidden_feature) -> (batch_size * seq_len, *hidden_feature)
                encoding = encoding.flatten(0, 1)
                pdparam = self.actor_layer(encoding)
                extrinsic_value = self.extrinic_critic_layer(encoding)
                intrinsic_value = self.intrinsic_value(encoding)
                
                return pdparam, extrinsic_value, intrinsic_value, next_hidden_state
        """
        raise NotImplementedError

class RNDNetwork(nn.Module):
    """
    Random Network Distillation (RND) network. 
    It constitutes of predictor and target networks. 
    The target network is determinsitic, which means it will be never updated.
    """
    
    @property
    @abstractmethod
    def predictor(self) -> nn.Module:
        """Predictor network."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def target(self) -> nn.Module:
        """Target network."""
        raise NotImplementedError
    
    def forward(self, next_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Compute both predicted feature and target feature.

        Args:
            next_obs (Tensor): next observation batch

        Returns:
            Tuple[Tensor, Tensor]: predicted feature, target feature
            
        ## Input/Output Details
        
        `out_features` depends on you.
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |next observation batch|`(batch_size, *obs_shape)`|
        
        Output:
        
        |Input|Shape|
        |:---|:---|
        |predicted feature|`(batch_size, out_features)`|
        |target feature|`(batch_size, out_features)`|
        """
        predicted_feature = self.predictor(next_obs)
        with torch.no_grad():
            target_feature = self.target(next_obs)
        return predicted_feature, target_feature
    
class HistroyRNDNetwork(nn.Module):
    """
    Random Network Distillation (RND) network with the action-observation history. 
    
    When POMDP, the actor-critic networks condition on the history instead of the true state. 
    This can be achived by using the reucrrent network and the network can learn the hidden state h. 
    You can use the hidden state to RND, but this is optional, so it's okay to use only the next observation. 
    Since RND doesn't use the recurrent layers, 
    you should use the hidden state by concatenating with the next observation.
    
    It constitutes of the predictor and target networks. 
    Both of them must have the same architectures.
    The target network is determinsitic, which means it will be never updated. 
    """
    
    def forward(self, next_obs: torch.Tensor, next_hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Compute both predicted feature and target feature. 
        You can use the hidden state by concatenating the next observation or the feature extracted from it with the hidden state. 

        Args:
            next_obs (Tensor): next observation batch
            next_hidden_state (Tensor): next hidden state batch with flattened features

        Returns:
            predicted feature (Tensor): predicted feature whose gradient flows
            target feature (Tensor): target feature whose gradient doesn't flow
            
        ## Input/Output Details
        
        The value of `out_features` depends on you.
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |next observation batch|`(batch_size, *obs_shape)`|
        |next hidden state batch|`(batch_size, D x num_layers x H)`|
        
        Output:
        
        |Input|Shape|
        |:---|:---|
        |predicted feature|`(batch_size, out_features)`|
        |target feature|`(batch_size, out_features)`|
        """
        raise NotImplementedError
    
class RecurrentActorCriticSharedRNDNetwork(RecurrentNetwork):
    """
    It constitutes of `RecurrentActorCriticSharedTwoValueNetwork` and `HistroyRNDNetwork`. 
    See details in each docs. 
    You don't need to implement `forward()` method.
    """
    
    @property
    @abstractmethod
    def actor_critic_net(self) -> RecurrentActorCriticSharedTwoValueNetwork:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def rnd_net(self) -> HistroyRNDNetwork:
        raise NotImplementedError
    
class SACNetwork(Network):
    """
    Soft Actor Critic (SAC) network.
    """
    
    @property
    @abstractmethod
    def v_net(self) -> VNetwork:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def q_net1(self) -> QNetwork:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def q_net2(self) -> QNetwork:
        raise NotImplementedError
    
    # @property
    # @abstractmethod
    # def actor(self) -> PolicyGradientNetwork:
    #     raise NotImplementedError
    
    def train_step(self, loss: torch.Tensor, grad_clip_max_norm: Optional[float], training_step: int):
        """It's not called from SAC agent."""
        pass
