from .register import torch2of, default_converter
from .proxy import (
    ProxySubmodule,
    proxy_class,
    replace_obj,
    map_args,
    replace_func,
    get_attr,
)
from ._globals import add_to_proxy_of_mds