from typing import NamedTuple, Optional

class REINFORCEConfig(NamedTuple):
    """
    REINFORCE configuration.

    Args:
        `gamma (float, optional)`: discount factor. Defaults to 0.99.
        `entropy_coef (float, optional)`: entropy multiplier used to compute loss. It adjusts exploration/exploitation balance. Defaults to 0.001.
        `grad_clip_max_norm (float | None, optional)`: maximum norm for the gradient clipping. Defaults to no gradient clipping.
    """
    gamma: float = 0.99
    entropy_coef: float = 0.001
    grad_clip_max_norm: Optional[float] = None
