from typing import NamedTuple, Optional

class PPOConfig(NamedTuple):
    """
    PPO configurations.

    Args:
        `training_freq (int)`: training frequency which is the number of time steps to gather experiences
        `epoch (int)`: number of using total gathered experiences to update parameters at each training frequency
        `mini_batch_size` (int): mini-batch size determines how many training steps at each epoch. The number of updates at each epoch equals to integer of `num_envs` x `training_freq` / `mini_batch_size`.
        `gamma (float, optional)`: discount factor. Defaults to 0.99.
        `lam (float, optional)`: regularization parameter which controls the balanace of Generalized Advantage Estimation (GAE) between bias and variance. Defaults to 0.95.
        `advantage_normalization (bool, optional)`: normalize advantage estimates across single mini batch. It may reduce variance and lead to stability, but does not seem to effect performance much. Defaults to False.
        `epsilon_clip (float, optional)`: clipping the probability ratio (pi_theta / pi_theta_old) to [1-eps, 1+eps]. Defaults to 0.2.
        `value_loss_coef (float, optional)`: state value loss (critic loss) multiplier. Defaults to 0.5.
        `entropy_coef (float, optional)`: entropy multiplier used to compute loss. It adjusts exploration/exploitation balance. Defaults to 0.001.
        `grad_clip_max_norm (float | None, optional)`: maximum norm for the gradient clipping. Defaults to no gradient clipping.
    """
    training_freq: int
    epoch: int
    mini_batch_size: int
    gamma: float = 0.99
    lam: float = 0.95
    advantage_normalization: bool = False
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    grad_clip_max_norm: Optional[float] = None