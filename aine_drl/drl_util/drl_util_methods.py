import torch.nn as nn
import torch
from aine_drl.network import Network

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
        
def copy_module(src_module: nn.Module, target_module: nn.Module):
    """
    Copy model weights from src to target.
    """
    target_module.load_state_dict(src_module.state_dict())

def polyak_update_module(src_module: nn.Module, target_module: nn.Module, src_ratio: float):
    assert src_ratio >= 0 and src_ratio <= 1
    for src_param, target_param in zip(src_module.parameters(), target_module.parameters()):
        target_param.data.copy_(src_ratio * src_param.data + (1.0 - src_ratio) * target_param.data)

def copy_network(src_net: Network, target_net: Network):
    """
    Copy model weights from src to target.
    """
    target_net.load_state_dict(src_net.state_dict())
    
def polyak_update_network(src_net: Network, target_net: Network, src_ratio: float):
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

def normalize(x: torch.Tensor, mask: bool | torch.Tensor = True) -> torch.Tensor:
    return (x - x[mask].mean()) / (x[mask].std() + 1e-8)
