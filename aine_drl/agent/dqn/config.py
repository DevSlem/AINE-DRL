from typing import NamedTuple

class DoubleDQNConfig(NamedTuple):
    """
    Double DQN configurations. 
    If both `replace_freq` and `polyak_ratio` are `None`, it uses `replace_freq` as `1`. If both of them are activated, it uses `replace_freq`.

    Args:
        `training_freq (int)`: training frequency which is the number of time steps to gather experiences
        `batch_size (int)`: size of experience batch from experience replay
        `capacity (int)`: number of experineces to be stored in experience replay
        `epoch (int)`: number of parameters updates at each `training_freq`
        `gamma (float, optional)`: discount factor. Defaults to 0.99.
        `replace_freq (int | None, optional)`: freqeuncy which totally replaces target network with update network. Defaults to None.
        `polyak_ratio (float | None, optional)`: smooth replace multiplier. `polyak_ratio` must be 0 < p <= 1. Defaults to None.
    """
    training_freq: int
    batch_size: int
    capacity: int
    epoch: int
    gamma: float = 0.99
    replace_freq: int | None = None
    polyak_ratio: float | None = None
