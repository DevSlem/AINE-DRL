from aine_drl.agent import Agent
from aine_drl.drl_algorithm import DQN, DQNSpec, TargetNetUpdateType
from aine_drl.drl_util import Clock
from aine_drl.util import except_dict_element
import aine_drl.policy as P
import aine_drl.util as util

class DQNAgent(Agent):
    def __init__(self, dqn_spec: DQNSpec, config: dict) -> None:
        training_config = config["training"]
        dqn_config = config["DQN"]
        target_net_update_type_config = dqn_config.get(
            "target_net_update_config",
            {"name": "replace"}
        )
        
        env_count = training_config["env_count"]
        
        if training_config['linear decay'] == 'linear ' 
        
        clock = Clock(env_count)
        drl_algorithm = DQN(
            dqn_spec,
            clock,
            dqn_config.get("gamma", 0.99),
            getattr(TargetNetUpdateType, target_net_update_type_config["name"]),
            **except_dict_element(target_net_update_type_config, "name")
        )
        
        policy_config = config["policy"]
        if policy_config["name"] == "EpsilonGreedyPolicy":
            epsilon_decay_config = policy_config["epsilon_decay"]
            epsilon_decay = getattr(util, epsilon_decay_config["name"])(**except_dict_element(epsilon_decay_config, "name"))
            policy = P.EpsilonGreedyPolicy(epsilon_decay)
            
        trajectory_config = config["trajectory"]
        trajectory = getattr(util, )
        
        super().__init__(drl_algorithm, policy, trajectory, clock, summary_freq)