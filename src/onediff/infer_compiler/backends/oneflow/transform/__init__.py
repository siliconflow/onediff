"""Module to convert PyTorch code to OneFlow."""
from .builtin_transform import (
    default_converter,
    get_attr,
    map_args,
    proxy_class,
    ProxySubmodule,
    torch2oflow,
)
from .custom_transform import register
from .manager import transform_mgr
