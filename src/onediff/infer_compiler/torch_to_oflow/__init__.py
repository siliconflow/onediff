"""Module to convert PyTorch code to OneFlow."""
import os
import warnings

from .register import torch2oflow, default_converter

from .proxy import (
    ProxySubmodule,
    proxy_class,
    replace_obj,
    map_args,
    replace_func,
    get_attr,
)

from ._globals import update_class_proxies, load_class_proxies_from_packages

if os.environ.get("ONEDIFF_SUPPRESS_WARNINGS", "1") == "1":
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)

