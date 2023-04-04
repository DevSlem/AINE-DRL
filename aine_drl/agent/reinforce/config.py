from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class REINFORCEConfig:
    """
    REINFORCE configuration.

    Docs: https://devslem.github.io/AINE-DRL/agent/reinforce#configuration
    """
    gamma: float = 0.99
    entropy_coef: float = 0.001
    device: str | None = None