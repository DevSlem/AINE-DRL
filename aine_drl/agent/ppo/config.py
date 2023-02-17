from typing import NamedTuple, Tuple

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
    
class RecurrentPPOConfig(NamedTuple):
    """
    Recurrent PPO configurations.

    Args:
        `training_freq (int)`: training frequency which is the number of time steps to gather experiences
        `epoch (int)`: number of using total experiences to update parameters at each training frequency
        `sequence_length (int)`: sequence length of recurrent network when training. trajectory is split by `sequence_length` unit. a value of `8` or greater are typically recommended.
        `num_sequences_per_step (int)`: number of sequences per train step, which are selected randomly
        `padding_value (float, optional)`: pad sequences to the value for the same `sequence_length`. Defaults to 0.
        `gamma (float, optional)`: discount factor. Defaults to 0.99.
        `lam (float, optional)`: regularization parameter which controls the balanace of Generalized Advantage Estimation (GAE) between bias and variance. Defaults to 0.95.
        `epsilon_clip (float, optional)`: clipping the probability ratio (pi_theta / pi_theta_old) to [1-eps, 1+eps]. Defaults to 0.2.
        `value_loss_coef (float, optional)`: state value loss (critic loss) multiplier. Defaults to 0.5.
        `entropy_coef (float, optional)`: entropy multiplier used to compute loss. It adjusts exploration/exploitation balance. Defaults to 0.001.
    """
    training_freq: int
    epoch: int
    sequence_length: int
    num_sequences_per_step: int
    padding_value: float = 0.0
    gamma: float = 0.99
    lam: float = 0.95
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001

class RecurrentPPORNDConfig(NamedTuple):
    """
    Recurrent PPO with RND configurations.

    Args:
        `training_freq (int)`: training frequency which is the number of time steps to gather experiences
        `epoch (int)`: number of using total experiences to update parameters at each training frequency
        `sequence_length (int)`: sequence length of recurrent network when training. trajectory is split by `sequence_length` unit. a value of `8` or greater are typically recommended.
        `num_sequences_per_step (int)`: number of sequences per train step, which are selected randomly
        `padding_value (float, optional)`: pad sequences to the value for the same `sequence_length`. Defaults to 0.
        `extrinsic_gamma (float, optional)`: discount factor of extrinsic reward. Defaults to 0.99.
        `intrinsic_gamma (float, optional)`: discount factor of intrinsic reward. Defaults to 0.99.
        `extrinsic_adv_coef (float, optional)`: multiplier to extrinsic advantage. Defaults to 1.0.
        `intrinsic_adv_coef (float, optional)`: multiplier to intrinsic advantage. Defaults to 1.0.
        `lam (float, optional)`: regularization parameter which controls the balanace of Generalized Advantage Estimation (GAE) between bias and variance. Defaults to 0.95.
        `epsilon_clip (float, optional)`: clipping the probability ratio (pi_theta / pi_theta_old) to [1-eps, 1+eps]. Defaults to 0.2.
        `value_loss_coef (float, optional)`: state value loss (critic loss) multiplier. Defaults to 0.5.
        `entropy_coef (float, optional)`: entropy multiplier used to compute loss. It adjusts exploration/exploitation balance. Defaults to 0.001.
        `exp_proportion_for_predictor (float, optional)`: proportion of experience used for training predictor to keep the effective batch size. Defaults to 0.25.
        `pre_normalization_step (int | None, optional)`: number of initial steps for initializing both observation and hidden state normalization. Defaults to no normalization.
        `obs_norm_clip_range (Tuple[float, float])`: observation normalization clipping range (min, max). Defaults to (-5.0, 5.0).
        `hidden_state_norm_clip_range (Tuple[float, float])`: hidden state normalization clipping range (min, max). Defaults to (-5.0, 5.0).
    """
    training_freq: int
    epoch: int
    sequence_length: int
    num_sequences_per_step: int
    padding_value: float = 0.0
    extrinsic_gamma: float = 0.99
    intrinsic_gamma: float = 0.99
    extrinsic_adv_coef: float = 1.0
    intrinsic_adv_coef: float = 1.0
    lam: float = 0.95
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    exp_proportion_for_predictor: float = 0.25
    pre_normalization_step: int | None = None
    obs_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
    hidden_state_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
