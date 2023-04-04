from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class A2CConfig:
    """
    A2C configurations.

    Docs: https://devslem.github.io/AINE-DRL/agent/a2c#configuration
    """
    n_steps: int
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    device: str | None = None