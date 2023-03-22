from abc import abstractmethod

import torch

from aine_drl.net import Network, RecurrentNetwork
from aine_drl.policy.policy import PolicyDistParam


class PPOSharedNetwork(Network):
    """
    Proximal Policy Optimization (PPO) shared network. 
    
    Note that since it uses the Actor-Critic architecure and the parameter sharing, 
    the encoding layer must be shared between Actor and Critic. 
    """
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> tuple[PolicyDistParam, torch.Tensor]:
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

class RecurrentPPOSharedNetwork(RecurrentNetwork):
    """
    Recurrent Proximal Policy Optimization (PPO) shared network.
    
    Since it uses the recurrent network, 
    you must consider the hidden state which can acheive the action-observation history.
    
    Note that since it uses the Actor-Critic architecure and the parameter sharing, 
    the encoding layer must be shared between Actor and Critic. 
    """
        
    @abstractmethod
    def forward(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor) -> tuple[PolicyDistParam, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution parameter (pdparam) 
        and state value using the recurrent network.
        
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
        |obs_seq|`(seq_batch_size, seq_len, *obs_shape)`|
        |hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |pdparam_seq|`*batch_shape` = `(seq_batch_size, seq_len)`, details in `PolicyDistParam` docs|
        |state_value_seq|`(seq_batch_size, seq_len, 1)`|
        |next_seq_hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Refer to the following explanation:
        
        * `seq_batch_size`: the number of independent sequences
        * `seq_len`: the length of each sequence
        * `num_layers`: the number of recurrent layers
        * `D`: 2 if bidirectional otherwise 1
        * `H`: the value depends on the type of the recurrent network
        
        When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
            
        ## Examples
        
        `forward()` method example when using LSTM::
        
            def forward(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor) -> tuple[aine_drl.PolicyDistParam, torch.Tensor, torch.Tensor]:
                # feed forward to the encoding layer
                # (seq_batch_size, seq_len, *obs_shape) -> (seq_batch_size * seq_len, *obs_shape)
                _, seq_len, _ = self.unpack_seq_shape(obs_seq)
                obs_batch = obs_seq.flatten(0, 1)
                encoded_batch = self.encoding_layer(obs_batch)
                
                # LSTM layer
                unpacked_hidden_state = self.unpack_lstm_hidden_state(hidden_state)
                # (seq_batch_size * seq_len, lstm_in_features) -> (seq_batch_size, seq_len, lstm_in_features)
                encoded_seq = encoded_batch.reshape(-1, seq_len, self.lstm_in_features)
                encoded_seq, unpacked_next_seq_hidden_state = self.lstm_layer(encoded_seq, unpacked_hidden_state)
                next_seq_hidden_state = self.pack_lstm_hidden_state(unpacked_next_seq_hidden_state)
                
                # actor-critic layer
                # (seq_batch_size, seq_len, D x H_out) -> (seq_batch_size * seq_len, D x H_out)
                encoded_batch = encoded_seq.flatten(0, 1)
                pdparam_batch = self.actor_layer(encoded_batch)
                state_value_batch = self.critic_layer(encoded_batch)
                
                # (seq_batch_size * seq_len, *shape) -> (seq_batch_size, seq_len, *shape)
                pdparam_seq = pdparam_batch.transform(lambda x: x.reshape(-1, seq_len, *x.shape[1:]))
                state_value_seq = state_value_batch.reshape(-1, seq_len, 1)
                
                return pdparam_seq, state_value_seq, next_seq_hidden_state
        """
        raise NotImplementedError
    
class RecurrentPPORNDNetwork(RecurrentNetwork):
    """
    Recurrent Proximal Policy Optimization (PPO) shared network with Random Network Distillation (RND).
    
    Since it uses the recurrent network, you must consider the hidden state which can acheive the action-observation history.
    
    Note that since PPO uses the Actor-Critic architecure and the parameter sharing, 
    the encoding layer must be shared between Actor and Critic. 
    Be careful not to share parameters between PPO and RND networks.
    
    RND uses extrinsic and intrinsic reward streams. 
    Each stream can be different episodic or non-episodic, and can have different discount factors. 
    RND constitutes of the predictor and target networks. 
    Both of them must have the same architectures but their initial parameters should not be the same.
    The target network is determinsitic, which means it will be never updated. 
    """
        
    @abstractmethod
    def forward_actor_critic(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor) -> tuple[PolicyDistParam, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution parameter (pdparam), 
        extinrisc state value and intrinsic state value using the recurrent network.
                
        It's recommended to set your recurrent network to `batch_first=True`.
        
        Args:
            obs_seq (Tensor): observation sequences
            hidden_state (Tensor): hidden states at the beginning of each sequence

        Returns:
            pdparam_seq (Tensor): policy distribution parameter (categorical probabilities) sequences
            ext_state_value_seq (Tensor): extrinsic state value sequences
            int_state_value_seq (Tensor): intrinsic state value sequences
            next_seq_hidden_state (Tensor): hidden state which will be used for the next sequence
        
        ## Input/Output Details
                
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs_seq|`(seq_batch_size, seq_len, *obs_shape)`|
        |hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |pdparam_seq|`*batch_shape` = `(seq_batch_size, seq_len)`, details in `PolicyDistParam` docs|
        |ext_state_value_seq|`(seq_batch_size, seq_len, 1)`|
        |int_state_value_seq|`(seq_batch_size, seq_len, 1)`|
        |next_seq_hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Refer to the following explanation:
        
        * `seq_batch_size`: the size of sequence batch
        * `seq_len`: the length of each sequence
        * `num_layers`: the number of recurrent layers
        * `D`: 2 if bidirectional otherwise 1
        * `H`: the value depends on the type of the recurrent network
        
        When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_rnd(self, next_obs: torch.Tensor, next_hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute both predicted feature and target feature. 
        You can use the hidden state by concatenating the next observation or the feature extracted from it with the hidden state. 

        Args:
            next_obs (Tensor): next observation batch
            next_hidden_state (Tensor): next hidden state batch with flattened features

        Returns:
            predicted_feature (Tensor): predicted feature whose gradient flows
            target_feature (Tensor): target feature whose gradient doesn't flow
            
        ## Input/Output Details
        
        The value of `out_features` depends on you.
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |next_obs|`(batch_size, *obs_shape)`|
        |next_hidden_state|`(batch_size, D x num_layers x H)`|
        
        Output:
        
        |Input|Shape|
        |:---|:---|
        |predicted_feature|`(batch_size, out_features)`|
        |target_feature|`(batch_size, out_features)`|
        """
        raise NotImplementedError
