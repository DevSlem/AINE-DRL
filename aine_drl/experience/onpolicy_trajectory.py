from aine_drl.experience import FixedCountTrajectory

class OnPolicyTrajectory(FixedCountTrajectory):
    
    def __init__(self, training_frequency: int, max_count_per_env: int, env_count: int = 1) -> None:
        super().__init__(max_count_per_env, env_count)
        self.frequency = training_frequency