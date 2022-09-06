_global_env_id = ""

def set_global_env_id(global_env: str):
    global _global_env_id
    _global_env_id = global_env
    
def get_global_env_id() -> str:
    global _global_env_id
    return _global_env_id
