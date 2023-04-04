from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DoubleDQNConfig:
    """
    Double DQN configurations. 
    
    Docs: https://devslem.github.io/AINE-DRL/agent/double-dqn.html#configuration
    """
    n_steps: int
    batch_size: int
    capacity: int
    epoch: int
    gamma: float = 0.99
    replace_freq: int | None = None
    polyak_ratio: float | None = None
    replay_buffer_device: str | None = None
    device: str | None = None