from dataclasses import dataclass

@dataclass(frozen=True)
class A2CConfig:
    """
    A2C configurations.

    Args:
        `n_steps (int)`: the number of time steps to collect experiences until training
        `gamma (float, optional)`: discount factor. Defaults to 0.99.
        `lam (float, optional)`: lambda or GAE regularization parameter. Defaults to 0.95.
        `value_loss_coef (float, optional)`: state value loss (critic loss) multiplier. Defaults to 0.5.
        `entropy_coef (float, optional)`: entropy multiplier. Defaults to 0.001.
    """
    n_steps: int
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
