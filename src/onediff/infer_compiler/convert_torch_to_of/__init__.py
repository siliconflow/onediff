import os

if os.environ.get("ONEDIFF_SUPPRESS_WARNINGS", "1") == "1":
    import warnings

    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)

from .register import torch2onef, default_converter
from .proxy import (
    ProxySubmodule,
    proxy_class,
    replace_obj,
    map_args,
    replace_func,
    get_attr,
)
from ._globals import update_class_proxies
