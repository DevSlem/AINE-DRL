from typing import Union, Tuple, NamedTuple, List, Optional
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence

def batch2perenv(batch: torch.Tensor, num_envs: int) -> torch.Tensor:
    """
    `(num_envs x n_steps, *shape)` -> `(num_envs, n_steps, *shape)`
    
    The input `batch` must be like the following example `Before`:
    
    `num_envs` = 2, `n_steps` = 3
    
    Before::
    
        [env1_step0, 
         env2_step0, 
         env1_step1, 
         env2_step1, 
         env1_step2, 
         env2_step2]
         
    After::
    
        [[env1_step0, env1_step1, env1_step2],
         [env2_step0, env2_step1, env2_step2]]
    
    """
    shape = batch.shape
    # scalar data (num_envs * n,)
    if len(shape) < 2:
        return batch.reshape(-1, num_envs).T
    # non-scalar data (num_envs * n, *shape)
    else:
        shape = (-1, num_envs) + shape[1:]
        return batch.reshape(shape).transpose(0, 1)

def perenv2batch(per_env: torch.Tensor) -> torch.Tensor:
    """
    `(num_envs, n_steps, *shape)` -> `(num_envs x n_steps, *shape)`
    
    The input `per_env` must be like the following example `Before`:
    
    `num_envs` = 2, `n_steps` = 3
         
    Before::
    
        [[env1_step0, env1_step1, env1_step2],
         [env2_step0, env2_step1, env2_step2]]
         
    After::
    
        [env1_step0, 
         env2_step0, 
         env1_step1, 
         env2_step1, 
         env1_step2, 
         env2_step2]
    """
    shape = per_env.shape
    # scalar data (num_envs, n,)
    if len(shape) < 3:
        return per_env.T.reshape(-1)
    # non-scalar data (num_envs, n, *shape)
    else:
        shape = (-1,) + shape[2:]
        return per_env.transpose(0, 1).reshape(shape)
    
class TruncatedSequenceGenerator:
    """
    ## Summary
    
    Truncated sequence generator from batch.

    Args:
        full_seq_len (int): full sequence length
        num_envs (int): number of environments
        n_steps (int): number of time steps
        padding_value (float, optional): padding value. Defaults to 0.
        
    ## Example
    
    ::
    
        full_seq_len, num_envs, n_steps = 4, 2, 7
        
        seq_generator = TruncatedSequenceGenerator(full_seq_len, num_envs, n_steps)
        
        hidden_state = torch.randn((num_envs, n_steps, 3, 8)) # (num_envs, n_steps, D x num_layers, H)
        obs = torch.randn((num_envs, n_steps, 3)) # (num_envs, n_steps, obs_features)
        terminated = torch.zeros((num_envs, n_steps))
        terminated[0, 2], terminated[0, 4] = 1, 1
        
        seq_generator.add(hidden_state, seq_len=1)
        seq_generator.add(obs)
        
        mask, seq_init_hidden_state, obs_seq = seq_generator.generate(terminated)
        seq_init_hidden_state.squeeze_(dim=1).swapaxes_(0, 1) # (D x num_layers, num_seq, H)
        
        >>> mask.shape, seq_init_hidden_state.shape, obs_seq.shape
        (torch.Size([5, 4]), torch.Size([3, 5, 8]), torch.Size([5, 4, 3]))
    """
    
    class __SeqInfo(NamedTuple):
        batch: torch.Tensor
        start_idx: int
        seq_len: int
    
    def __init__(self, 
                 full_seq_len: int, 
                 num_envs: int, 
                 n_steps: int,
                 padding_value: float = 0.0) -> None:
        if full_seq_len <= 0:
            raise ValueError(f"full_seq_len must be greater than 0, but got {full_seq_len}")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be greater than 0, but got {num_envs}")
        if n_steps <= 0: 
            raise ValueError(f"n_steps must be greater than 0, but got {n_steps}")
                
        self._seq_infos: List[TruncatedSequenceGenerator.__SeqInfo] = []
        self._full_seq_len = full_seq_len
        self._num_envs = num_envs
        self._n_steps = n_steps
        self._padding_value = padding_value
    
    def add(self, batch: torch.Tensor, start_idx: int = 0, seq_len: int = 0):
        """
        Add a batch to make a truncated sequence.

        Args:
            batch (Tensor): `(num_envs, n_steps, *shape)`
            start_idx (int, optional): start index of each sequence. Defaults to the sequence start point.
            seq_len (int, optional): the length of each sequence. Defaults to `full_seq_len` - `start_idx` which is from start to end of the sequence.
            
        Raises:
            ValueError: when `start_idx` or `seq_len` is out of sequence range
        """
        if len(batch.shape) < 2 or batch.shape[0] != self._num_envs or batch.shape[1] != self._n_steps:
            raise ValueError(f"batch must be (num_envs, n_steps, *shape), but got {batch.shape}")
        
        # pre-process start_idx
        if start_idx < -self._full_seq_len or start_idx >= self._full_seq_len:
            raise ValueError(f"start_idx={start_idx} is out of sequence range [{-self._full_seq_len}, {self._full_seq_len - 1}].")
        if start_idx < 0:
            start_idx += self._full_seq_len
            
        # pre-process seq_len
        len_to_end = self._full_seq_len - start_idx
        if seq_len <= -len_to_end or seq_len > len_to_end:
            raise ValueError(f"seq_len={seq_len} is out of sequence range [{-len_to_end + 1}, {len_to_end}].")
        if seq_len <= 0:
            seq_len += len_to_end
            
        # add sequence info
        self._seq_infos.append(TruncatedSequenceGenerator.__SeqInfo(batch, start_idx, seq_len))

    def generate(self, interrupted_binary_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Generate truncated sequences.

        Args:
            interrupted_binary_mask (Tensor | None, optional): interrupted binary mask `(num_envs, n_steps)`. 
            it's generally terminated tensor. Defaults to no interrupting.

        Raises:
            ValueError: No batch is added before calling this method.

        Returns:
            mask (Tensor): `(num_seq, full_seq_len)`
            generated seuqences (Tensor, ...): `(num_seq, seq_len, *shape)`
        """
        if len(self._seq_infos) == 0:
            raise ValueError("No batch is added before calling this method.")
        
        # field cashing
        seq_infos = self._seq_infos
        num_envs = self._num_envs
        n_steps = self._n_steps
        full_seq_len = self._full_seq_len
        
        # add the mask at the beginning of the list
        mask = torch.ones((num_envs, n_steps))
        seq_infos.insert(0, TruncatedSequenceGenerator.__SeqInfo(mask, 0, full_seq_len))
        
        stacked_batches = [[] for _ in range(len(seq_infos))]
        
        for env_id in range(num_envs):
            seq_start_t = 0
            interrupted_idx = 0
            
            interrupted_time_steps = torch.tensor([]) if interrupted_binary_mask is None else torch.where(interrupted_binary_mask[env_id] > 0.5)[0]
            
            while seq_start_t < n_steps:
                # when passing the last time step, it will be zero padded
                seq_end_t = min(seq_start_t + full_seq_len, n_steps)
                
                # if interupped in the middle of the sequence, it will be zero padded
                if interrupted_idx < len(interrupted_time_steps) and interrupted_time_steps[interrupted_idx] < seq_end_t:
                    seq_end_t = interrupted_time_steps[interrupted_idx].item() + 1
                    interrupted_idx += 1
                
                for i, seq_info in enumerate(seq_infos):
                    # determine the sequence range of the current sequence
                    current_seq_start_t = seq_start_t + seq_info.start_idx
                    current_seq_end_t = min(current_seq_start_t + seq_info.seq_len, seq_end_t)
                    current_seq_time_steps = torch.arange(current_seq_start_t, current_seq_end_t)
                    stacked_batches[i].append(seq_info.batch[env_id, current_seq_time_steps])
                    
                seq_start_t = seq_end_t
                
        # pad all sequences
        padded_sequences = []
        for stacked_batch in stacked_batches:
            padded_sequences.append(pad_sequence(stacked_batch, batch_first=True, padding_value=self._padding_value))

        # convert the float mask to boolean
        padded_sequences[0] = padded_sequences[0] > 0.5
        return tuple(padded_sequences)
        
def copy_network(src_net: nn.Module, target_net: nn.Module):
    """
    Copy model weights from src to target.
    """
    target_net.load_state_dict(src_net.state_dict())

def polyak_update(src_net: nn.Module, target_net: nn.Module, src_ratio: float):
    assert src_ratio >= 0 and src_ratio <= 1
    for src_param, target_param in zip(src_net.parameters(), target_net.parameters()):
        target_param.data.copy_(src_ratio * src_param.data + (1.0 - src_ratio) * target_param.data)

def compute_return(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute return.
    
    Args:
        rewards (Tensor): whose shape is `(episode_len,)`
        gamma (float): discount factor
        
    Returns:
        Tensor: return whose shape is `(episode_len,)`
    """
    returns = torch.empty_like(rewards)
    G = 0 # return at the time step t
    for t in reversed(range(len(returns))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

def compute_gae(v_preds: torch.Tensor, 
                rewards: torch.Tensor, 
                terminateds: torch.Tensor,
                gamma: float,
                lam: float) -> torch.Tensor:
    """
    Compute generalized advantage estimation (GAE) during n-step transitions. See details in https://arxiv.org/abs/1506.02438.

    Args:
        v_preds (Tensor): predicted value batch whose shape is (num_envs, n+1), 
        which means the next state value of final transition must be included
        rewards (Tensor): reward batch whose shape is (num_envs, n)
        terminateds (Tensor): terminated batch whose shape is (num_envs, n)
        gamma (float): discount factor
        lam (float): lambda which controls the balanace between bias and variance

    Returns:
        Tensor: GAE whose shape is (num_envs, n)
    """
    
    n_step = rewards.shape[1]
    gaes = torch.empty_like(rewards)
    discounted_gae = 0.0 # GAE at time step t+n
    not_terminateds = 1 - terminateds
    delta = rewards + not_terminateds * gamma * v_preds[:, 1:] - v_preds[:, :-1]
    discount_factor = gamma * lam
    
    # compute GAE
    for t in reversed(range(n_step)):
        discounted_gae = delta[:, t] + not_terminateds[:, t] * discount_factor * discounted_gae
        gaes[:, t] = discounted_gae
     
    return gaes

def normalize(x: torch.Tensor, mask: Union[bool, torch.Tensor] = True) -> torch.Tensor:
    return (x - x[mask].mean()) / (x[mask].std() + 1e-8)
