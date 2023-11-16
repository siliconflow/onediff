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

if os.environ.get("ONEDIFF_DEBUG", "0") != "1":
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
