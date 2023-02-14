from typing import Tuple
from abc import abstractmethod
from aine_drl.network import Network, RecurrentNetwork
from aine_drl.policy.policy_distribution import PolicyDistParam
import torch

class PPOSharedNetwork(Network[torch.Tensor]):
    """
    Proximal Policy Optimization (PPO) shared network. 
    
    Note that since it uses the Actor-Critic architecure and the parameter sharing, 
    the encoding layer must be shared between Actor and Critic. 
    Therefore, single loss that is the sum of the actor and critic losses will be input.
    
    Generic type `T` is `Tensor`.
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> Tuple[PolicyDistParam, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution parameter (pdparam) and state value.

        Args:
            obs (Tensor): observation batch

        Returns:
            pdparam (PolicyDistParam): policy distribution parameter
            state_value (Tensor): state value V(s)
            
        ## Input/Output Details
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs|`(batch_size, *obs_shape)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |pdparam|`*batch_shape` = `(batch_size,)`, details in `PolicyDistParam` docs|
        |state_value|`(batch_size, 1)`|
        """
        raise NotImplementedError

class RecurrentPPOSharedNetwork(RecurrentNetwork[torch.Tensor]):
    """
    Recurrent Proximal Policy Optimization (PPO) shared network.
    
    Note that since it uses the Actor-Critic architecure and the parameter sharing, 
    the encoding layer must be shared between Actor and Critic. 
    Therefore, single loss that is the sum of the actor and critic losses will be input.
    
    Since it uses the recurrent network, you must consider the hidden state which can acheive the action-observation history.
    
    Generic type `T` is `Tensor`.
    """
        
    @abstractmethod
    def forward(self, 
                obs_seq: torch.Tensor, 
                hidden_state: torch.Tensor) -> Tuple[PolicyDistParam, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution parameter (pdparam) and state value using the recurrent network.
        
        It's recommended to set your recurrent network to `batch_first=True`.

        Args:
            obs_seq (Tensor): observation sequences
            hidden_state (Tensor): hidden states at the beginning of each sequence

        Returns:
            pdparam_seq (PolicyDistParam): policy distribution parameter sequences
            state_value_seq (Tensor): state value sequences
            next_seq_hidden_state (Tensor): hidden state which will be used for the next sequence
            
        ## Input/Output Details
            
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs_seq|`(num_seq, seq_len, *obs_shape)`|
        |hidden_state|`(D x num_layers, num_seq, H)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |pdparam_seq|`*batch_shape` = `(num_seq, seq_len)`, details in `PolicyDistParam` docs|
        |state_value_seq|`(num_seq, seq_len, 1)`|
        |next_seq_hidden_state|`(D x num_layers, num_seq, H)`|
        
        Refer to the following explanation:
        
        * `num_seq`: the number of independent sequences
        * `seq_len`: the length of each sequence
        * `num_layers`: the number of recurrent layers
        * `D`: 2 if bidirectional otherwise 1
        * `H`: the value depends on the type of the recurrent network
        
        When you use LSTM, `H` = `H_out x 2`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
            
        ## Examples
        
        `forward()` method example when using LSTM::
        
            def forward(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[aine_drl.PolicyDistParam, torch.Tensor, torch.Tensor]:
                # feed forward to the encoding layer
                # (num_seq, seq_len, *obs_shape) -> (num_seq * seq_len, *obs_shape)
                _, seq_len, _ = self.unpack_seq_shape(obs_seq)
                obs_batch = obs_seq.flatten(0, 1)
                encoded_batch = self.encoding_layer(obs_batch)
                
                # LSTM layer
                unpacked_hidden_state = self.unpack_lstm_hidden_state(hidden_state)
                # (num_seq * seq_len, lstm_in_features) -> (num_seq, seq_len, lstm_in_features)
                encoded_seq = encoded_batch.reshape(-1, seq_len, self.lstm_in_features)
                encoded_seq, unpacked_next_seq_hidden_state = self.lstm_layer(encoded_seq, unpacked_hidden_state)
                next_seq_hidden_state = self.pack_lstm_hidden_state(unpacked_next_seq_hidden_state)
                
                # actor-critic layer
                # (num_seq, seq_len, D x H_out) -> (num_seq * seq_len, D x H_out)
                encoded_batch = encoded_seq.flatten(0, 1)
                pdparam_batch = self.actor_layer(encoded_batch)
                state_value_batch = self.critic_layer(encoded_batch)
                
                # (num_seq * seq_len, *shape) -> (num_seq, seq_len, *shape)
                pdparam_seq = pdparam_batch.transform(lambda x: x.reshape(-1, seq_len, *x.shape[1:]))
                state_value_seq = state_value_batch.reshape(-1, seq_len, 1)
                
                return pdparam_seq, state_value_seq, next_seq_hidden_state
        """
        raise NotImplementedError