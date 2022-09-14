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
