from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    """
    PPO configurations.

    Docs: https://devslem.github.io/AINE-DRL/agent/ppo.html#configuration
    """
    n_steps: int
    epoch: int
    mini_batch_size: int
    gamma: float = 0.99
    lam: float = 0.95
    advantage_normalization: bool = False
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    device: str | None = None
    
@dataclass(frozen=True)
class RecurrentPPOConfig:
    """
    Recurrent PPO configurations.

    Docs: https://devslem.github.io/AINE-DRL/agent/recurrent-ppo.html#configuration
    """
    n_steps: int
    epoch: int
    seq_len: int
    seq_mini_batch_size: int
    padding_value: float = 0.0
    gamma: float = 0.99
    lam: float = 0.95
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    device: str | None = None
    
@dataclass(frozen=True)
class PPORNDConfig:
    """
    PPO with RND configurations.
    
    Docs: https://devslem.github.io/AINE-DRL/agent/ppo-rnd.html#configuration
    """
    
    n_steps: int
    epoch: int
    mini_batch_size: int
    ext_gamma: float = 0.999
    int_gamma: float = 0.99
    ext_adv_coef: float = 1.0
    int_adv_coef: float = 1.0
    lam: float = 0.95
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    rnd_pred_exp_proportion: float = 0.25
    init_norm_steps: int | None = 50
    obs_norm_clip_range: tuple[float, float] = (-5.0, 5.0)
    device: str | None = None

@dataclass(frozen=True)
class RecurrentPPORNDConfig:
    """
    Recurrent PPO with RND configurations.

    Docs: https://devslem.github.io/AINE-DRL/agent/recurrent-ppo-rnd.html#configuration
    """
    n_steps: int
    epoch: int
    seq_len: int
    seq_mini_batch_size: int
    padding_value: float = 0.0
    ext_gamma: float = 0.999
    int_gamma: float = 0.99
    ext_adv_coef: float = 1.0
    int_adv_coef: float = 1.0
    lam: float = 0.95
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    rnd_pred_exp_proportion: float = 0.25
    init_norm_steps: int | None = 50
    obs_norm_clip_range: tuple[float, float] = (-5.0, 5.0)
    hidden_state_norm_clip_range: tuple[float, float] = (-5.0, 5.0)
    device: str | None = None
