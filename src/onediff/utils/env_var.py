import os
from typing import Optional


def parse_boolean_from_env(env_var, default_value=None):
    env_var = os.getenv(env_var)
    if env_var is None:
        return default_value
    env_var = env_var.lower()
    return env_var in ("1", "true", "yes", "on", "y")


def set_boolean_env_var(env_var: str, val: Optional[bool]):
    if val is None:
        os.environ.pop(env_var, None)
    else:
        os.environ[env_var] = "1" if val else "0"


def parse_integer_from_env(env_var, default_value=None):
    env_var = os.getenv(env_var)
    if env_var is None:
        return default_value
    return int(env_var)


def set_integer_env_var(env_var: str, val: Optional[int]):
    if val is None:
        os.environ.pop(env_var, None)
    else:
        os.environ[env_var] = str(int(val))
