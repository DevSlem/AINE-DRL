r"""
`agent` module contains the `Agent` abstract class and provides a set of DRL agents.
"""

from .agent import BehaviorType, Agent, BehaviorScope
# DRL Agents
from .dqn import *
from .reinforce import *
from .a2c import *
from .ppo import *
