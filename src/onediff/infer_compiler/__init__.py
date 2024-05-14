import os
import torch

from .core import *
from .utils import set_default_env_vars
from .utils.options import CompileOptions
from .utils.options import _GLOBAL_compile_options as compile_options


set_default_env_vars()
