import torch
import torch.nn.functional as F

def true_return(
    reward: torch.Tensor,
    gamma: float
) -> torch.Tensor:
    """
    Compute true return.

    Args:
        reward (Tensor): `(episode_len,)`
        gamma (float): discount factor

    Returns:
        return (Tensor): `(episode_len,)`
    """
    ret = torch.empty_like(reward)
    G = 0 # return at the time step t
    for t in reversed(range(len(ret))):
        G = reward[t] + gamma * G
        ret[t] = G
    return ret

def gae(
    state_value: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    lam: float
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE) during n-step transitions. 
    
    See details in https://arxiv.org/abs/1506.02438.

    Args:
        state_value (Tensor): state value `(num_envs, n_steps + 1)`, 
        which includes the next state value of final transition
        reward (Tensor): `(num_envs, n_steps)`
        terminated (Tensor): `(num_envs, n_steps)`
        gamma (float): discount factor
        lam (float): lambda or bias-variance trade-off parameter

    Returns:
        GAE (Tensor): `(num_envs, n_steps)`
    """
    n_step = reward.shape[1]
    advantage = torch.empty_like(reward)
    discounted_gae = 0.0 # GAE at time step t+n
    not_terminated = 1 - terminated
    delta = reward + not_terminated * gamma * state_value[:, 1:] - state_value[:, :-1]
    discount_factor = gamma * lam
    
    # compute GAE
    for t in reversed(range(n_step)):
        discounted_gae = delta[:, t] + not_terminated[:, t] * discount_factor * discounted_gae
        advantage[:, t] = discounted_gae
     
    return advantage

def bellman_value_loss(
    predicted_state_value: torch.Tensor,
    target_state_value: torch.Tensor
) -> torch.Tensor:
    """
    Bellman value loss which is L2 loss.

    Args:
        predicted_state_value (Tensor): `(batch_size, 1)`
        target_state_value (torch.Tensor): `(batch_size, 1)`

    Returns:
        loss (Tensor): scalar value
    """
    return F.mse_loss(predicted_state_value, target_state_value)

def reinforce_loss(
    ret: torch.Tensor, 
    action_log_prob: torch.Tensor,
    baseline: bool = True
) -> torch.Tensor:
    """
    REINFORCE loss according to the policy gradient theorem.
    
    It uses mean loss (but not sum loss).

    Args:
        ret (Tensor): return `(episode_len,)`
        action_log_prob (torch.Tensor): `(episode_len,)`

    Returns:
        loss (Tensor): scalar value
    """
    eps = 1e-8
    if baseline:
        ret = (ret - ret.mean()) / (ret.std() + eps)
    return -(ret * action_log_prob).mean()