"""Module to convert PyTorch code to OneFlow."""
import os
import warnings

from .manager import transform_mgr
from .builtin_transform import torch2oflow, default_converter
from .custom_transform import register

from .builtin_transform import (
    ProxySubmodule,
    proxy_class,
    replace_obj,
    map_args,
    replace_func,
    get_attr,
)

if transform_mgr.debug_mode:
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
