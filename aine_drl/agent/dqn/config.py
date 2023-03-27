from dataclasses import dataclass


@dataclass(frozen=True)
class DoubleDQNConfig:
    """
    Double DQN configurations. 
    If both `replace_freq` and `polyak_ratio` are `None`, it uses `replace_freq` as `1`. If both of them are activated, it uses `replace_freq`.

    Args:
        `n_steps (int)`: the number of time steps to collect experiences until training
        `batch_size (int)`: the size of experience batch sampled from experience replay
        `capacity (int)`: the number of experineces to be stored in experience replay
        `epoch (int)`: the number of parameter updates at each training
        `gamma (float, optional)`: discount factor. Defaults to 0.99.
        `replace_freq (int | None, optional)`: freqeuncy which replaces target network parameters with update network one. Defaults to None.
        `polyak_ratio (float | None, optional)`: smooth replace multiplier. `polyak_ratio` must be 0 < p <= 1. Defaults to None.
    """
    n_steps: int
    batch_size: int
    capacity: int
    epoch: int
    gamma: float = 0.99
    replace_freq: int | None = None
    polyak_ratio: float | None = None
    replay_buffer_device: str = "auto"