import torch.nn as nn
import torch

def copy_network(src_net: nn.Module, target_net: nn.Module):
    """
    Copy model weights from src to target.
    """
    target_net.load_state_dict(src_net.state_dict())

def polyak_update(src_net: nn.Module, target_net: nn.Module, src_ratio: float):
    assert src_ratio >= 0 and src_ratio <= 1
    for src_param, target_param in zip(src_net.parameters(), target_net.parameters()):
        target_param.data.copy_(src_ratio * src_param.data + (1.0 - src_ratio) * target_param.data)

def calc_returns(rewards: torch.Tensor, terminateds: torch.Tensor, gamma: float):
    """
    Calculates returns. `rewards`, `terminateds` can be multiple episodes but must be flattend.
    """
    returns = torch.empty_like(rewards)
    G = 0 # return at time step t
    not_terminateds = 1 - terminateds
    for t in reversed(range(len(returns))):
        G = rewards[t] + not_terminateds[t] * gamma * G
        returns[t] = G
    return returns

def calc_gae(rewards: torch.Tensor, terminateds: torch.Tensor, v_preds: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """
    Calculates Generalized Advantage Estimations. See more details in https://arxiv.org/pdf/1506.02438.pdf.

    Args:
        rewards (torch.Tensor): rewards tensor
        terminateds (torch.Tensor): terminated
        v_preds (torch.Tensor): estimated state value which contains the next state value of the last transition
        gamma (float): discount factor
        lam (float): lambda which controls the balanace between bias and variance

    Returns:
        torch.Tensor: GAE
    """
    T = len(rewards)
    assert T + 1 == len(v_preds), "v_preds parameter must contain the value of the final next state."
    gaes = torch.empty_like(rewards)
    discounted_gae = 0 # GAE at time step t
    not_terminateds = 1 - terminateds
    delta = rewards + not_terminateds * gamma * v_preds[1:] - v_preds[:-1]
    discount_factor = gamma * lam
    for t in reversed(range(T)):
        discounted_gae = delta[t] + not_terminateds[t] * discount_factor * discounted_gae
        gaes[t] = discounted_gae
    return gaes
