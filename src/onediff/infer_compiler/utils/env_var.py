import os


def parse_boolean_from_env(env_var, default_value):
    env_var = os.getenv(env_var)
    if env_var is None:
        return default_value
    env_var = env_var.lower()
    return (
        env_var == "1"
        or env_var == "true"
        or env_var == "yes"
        or env_var == "on"
        or env_var == "y"
    )


def set_boolean_env_var(env_var: str, val: bool):
    os.environ[env_var] = str(val)
