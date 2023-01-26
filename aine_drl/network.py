from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Any, Iterator
from aine_drl.policy.policy_distribution import PolicyDistributionParameter
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def forward(self, x: torch.Tensor) -> PolicyDistributionParameter:
        out = self.layer(x)
        discrete_pdparams = list(torch.split(out, self.num_discrete_actions, dim=1))
        
        if not self.is_logits:
            for i in range(len(discrete_pdparams)):
                discrete_pdparams[i] = F.softmax(discrete_pdparams[i], dim=1)
        
        return PolicyDistributionParameter.new(discrete_pdparams=discrete_pdparams)
    
class GaussianContinuousActionLayer(nn.Module):
    """
    Linear layer for the continuous action type.

    Args:
        in_features (int): number of input features
        num_continuous_actions (int): number of continuous actions which equals to `num_continuous_branches`
    """
    
    def __init__(self, in_features: int, 
                 num_continuous_actions: int, 
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
        
        self.layer = nn.Linear(
            in_features,
            self.num_continuous_actions * 2,
            bias,
            device,
            dtype
        )
        
    def forward(self, x: torch.Tensor) -> PolicyDistributionParameter:
        out = self.layer(x)
        continuous_pdparams = list(torch.split(out, 2, dim=1))
        return PolicyDistributionParameter.new(continuous_pdparams=continuous_pdparams)
    
class Network(nn.Module, ABC):
    """
    AINE-DRL network abstract class.
    """
    
    @abstractmethod
    def train_step(self, 
                   loss: torch.Tensor,
                   grad_clip_max_norm: Optional[float],
                   training_step: int):
        """
        Gradient step for training.

        Args:
            loss (Tensor): computed loss
            grad_clip_max_norm (float | None): maximum norm for the gradient clipping
            training_step (int): current training step
        """
        raise NotImplementedError
        
    def basic_train_step(self,
                          loss: torch.Tensor,
                          optimizer: torch.optim.Optimizer,
                          grad_clip_max_norm: Optional[float]):
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_max_norm)
        optimizer.step()

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
    
class PolicyGradientNetwork(Network):
    """
    Policy gradient network.
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> PolicyDistributionParameter:
        """
        Compute policy distribution parameters whose shape is `(batch_size, ...)`. \\
        `batch_size` is `num_envs` x `n-step`. \\
        When the action type is discrete, policy distribution is generally logits or soft-max distribution. \\
        When the action type is continuous, it's generally mean and standard deviation of gaussian distribution.

        Args:
            obs (Tensor): observation of state whose shape is `(batch_size, *obs_shape)`

        Returns:
            PolicyDistributionParameter: policy distribution parameter
        """
        raise NotImplementedError

class ActorCriticSharedNetwork(Network):
    """
    Actor critic network.
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> Tuple[PolicyDistributionParameter, torch.Tensor]:
        """
        Compute policy distribution parameters whose shape is `(batch_size, ...)`, 
        state value whose shape is `(batch_size, 1)`. \\
        `batch_size` is `num_envs` x `n-step`. \\
        When the action type is discrete, policy distribution is generally logits or soft-max distribution. \\
        When the action type is continuous, it's generally mean and standard deviation of gaussian distribution.

        Args:
            obs (Tensor): observation of state whose shape is `(batch_size, *obs_shape)`

        Returns:
            Tuple[PolicyDistributionParameter, Tensor]: policy distribution parameter, state value
        """
        raise NotImplementedError

class QValueNetwork(Network):
    """
    Action value Q network.
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> PolicyDistributionParameter:
        """
        Compute action value Q.  \\
        Note that it only works to discrete action type. 
        So, you must set only `PolicyDistributionParameter.discrete_pdparams` which is action values. \\
        `batch_size` is `num_envs` x `n-step`.
        
        Args:
            obs (Tensor): observation of state whose shape is `(batch_size, *obs_shape)`
            
        Returns:
            PolicyDistributionParameter: discrete action value
        """
        raise NotImplementedError

class RecurrentNetwork(Network):
    """
    Standard recurrent neural network (RNN).
    """
    @staticmethod
    def pack_lstm_hidden_state(lstm_hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """`(D x num_layers, batch_size, H_out) x 2` -> `(D x num_layers, batch_size, H_out x 2)`"""
        return torch.cat(lstm_hidden_state, dim=2)
    
    @staticmethod
    def unpack_lstm_hidden_state(lstm_hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(D x num_layers, batch_size, H_out x 2)` -> `(D x num_layers, batch_size, H_out) x 2`"""
        lstm_hidden_state = lstm_hidden_state.split(lstm_hidden_state.shape[2] // 2, dim=2)  # type: ignore
        return (lstm_hidden_state[0].contiguous(), lstm_hidden_state[1].contiguous())
    
    @abstractmethod
    def hidden_state_shape(self, batch_size: int) -> Tuple[int, ...]:
        """
        Returns recurrent hidden state shape. 
        When you use LSTM, its shape is `(D x num_layers, batch_size, H_out x 2)`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, its shape is `(D x num_layers, batch_size, H_out)`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html. 
        """
        raise NotImplementedError
    
class RecurrentActorCriticSharedNetwork(RecurrentNetwork):
    """Recurrent actor critic shared network."""
        
    @abstractmethod
    def forward(self, 
                obs: torch.Tensor, 
                hidden_state: torch.Tensor) -> Tuple[PolicyDistributionParameter, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Compute policy distribution parameters whose shape is `(batch_size, ...)`, 
        state value whose shape is `(batch_size, 1)` and next recurrent hidden state. \\
        `batch_size` is flattened sequence batch whose shape is `sequence_batch_size` x `sequence_length`. \\
        It's recommended to set recurrent layer to `batch_first=True`. \\
        When the action type is discrete, policy distribution is generally logits or soft-max distribution. \\
        When the action type is continuous, it's generally mean and standard deviation of gaussian distribution. \\
        Recurrent hidden state is typically concatnated tensor of LSTM tuple (h, c) or GRU hidden state tensor.

        Args:
            obs (Tensor): observation of state whose shape is `(sequence_batch_size, sequence_length, *obs_shape)`
            hidden-state (Tensor): whose shape is `(max_num_layers, sequence_batch_size, out_features)`

        Returns:
            Tuple[PolicyDistributionParameter, Tensor, Tensor]: policy distribution parameter, state value, recurrent hidden state
            
        ## Examples
        
        `forward()` method example when using LSTM::
        
            def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[aine_drl.PolicyDistributionParameter, torch.Tensor, torch.Tensor]:
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
                v_pred = self.critic_layer(encoding)
                
                return pdparam, v_pred, next_hidden_state
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
                obs: torch.Tensor, 
                hidden_state: torch.Tensor) -> Tuple[PolicyDistributionParameter, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to estimate policy distribution parameter (pdparam), extrinsic state value, intrinsic state value using the recurrent layer.\\
        When the action type is discrete, pdparam is generally logits or soft-max distribution. \\
        When the action type is continuous, it's generally mean and standard deviation of gaussian distribution.
        
        Args:
            obs (Tensor): observation sequences
            hidden_state (Tensor): recurrent hidden state at the first time step of the sequences

        Returns:
            Tuple[PolicyDistributionParameter, Tensor, Tensor, Tensor]: policy distribution parameter, extrinsic state value, intrinsic state value, next recurrent hidden state
        
        ## Input/Output Details
        
        `batch_size` is flattened sequence batch whose shape is `sequence_batch_size` x `sequence_length`.
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |observation sequences|`(sequence_batch_size, sequence_length, *obs_shape)`|
        |hidden state|`(max_num_layers, sequence_batch_size, out_features)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |policy distribution parameter|details in `PolicyDistributionParameter`|
        |extrinsic state value|`(batch_size, 1)`|
        |intrinsic state value|`(batch-size, 1)`|
        |next hidden state|`(max_num_layers, sequence_batch_size, out_features)`|
        
        It's recommended to set the recurrent layer to `batch_first=True`. \\
        Hidden state is typically concatnated tensor of LSTM tuple (h, c) or GRU hidden state tensor.
            
        ## Examples
        
        `forward()` method example when using LSTM::
        
            def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[aine_drl.PolicyDistributionParameter, torch.Tensor, torch.Tensor]:
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
    Random Network Distillation (RND) Network. 
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
    
class RecurrentActorCriticSharedRNDNetwork(RecurrentNetwork):
    """
    It constitutes of `RecurrentActorCriticSharedTwoValueNetwork` and `RNDNetwork`. 
    See details in each docs. 
    You don't need to implement `forward()` method.
    """
    
    @property
    @abstractmethod
    def actor_critic_net(self) -> RecurrentActorCriticSharedTwoValueNetwork:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def rnd_net(self) -> RNDNetwork:
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
    
    @property
    @abstractmethod
    def actor(self) -> PolicyGradientNetwork:
        raise NotImplementedError
    
    def train_step(self, loss: torch.Tensor, grad_clip_max_norm: Optional[float], training_step: int):
        """It's not called from SAC agent."""
        pass
