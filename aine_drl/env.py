_env_id = ""

def set_env_id(env_id: str):
    global _env_id
    _env_id = env_id
    
def get_env_id() -> str:
    global _env_id
    return _env_id
