from .register import torch2of, default_converter
from .proxy import (
    ProxySubmodule,
    replace_obj,
    map_args,
    replace_func,
    get_attr,
    get_full_class_name,
)
from ._globals import add_to_proxy_of_mds
