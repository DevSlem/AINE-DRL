import yaml

class ConfigManager:
    def __init__(self, config_dir: str) -> None:
        self._config = ConfigManager.import_config(config_dir)
        
        keys = self._config.keys()
        assert len(keys) == 1
        self._env_id = list(keys)[0]
        
    @property
    def env_id(self) -> str:
        return self._env_id
        
    @property
    def env_config(self) -> dict:
        return self._config[self.env_id]
        
    @staticmethod
    def import_config(config_dir: str) -> dict:
        with open(config_dir) as f:
            config = yaml.load(f, yaml.FullLoader)
        
        return config