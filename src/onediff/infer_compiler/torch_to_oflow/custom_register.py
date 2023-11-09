"""A module for registering custom torch2oflow functions and classes."""
from pathlib import Path
import inspect

from ..import_tools import print_yellow, print_green, get_mock_cls_name
from .register import torch2oflow
from ._globals import update_class_proxies

__all__ = ["register_torch2of_class", "register_custom_torch2of_func"]


def register_torch2of_class(cls: type, replacement: type, verbose=True):
    try:
        key = get_mock_cls_name(cls)
        update_class_proxies({key: replacement}, verbose=verbose)

    except Exception as e:
        print_yellow(f"Cannot register {cls=} {replacement=}. {e=}")


def register_custom_torch2of_func(func, first_param_type=None, verbose=True):
    if first_param_type is None:
        params = inspect.signature(func).parameters
        first_param_type = params[list(params.keys())[0]].annotation
        if first_param_type == inspect._empty:
            print_yellow(f"Cannot register {func=} {first_param_type=}.")
    try:
        torch2oflow.register(first_param_type)(func)
        if verbose:
            print_green(f"Register {func=} {first_param_type=}")
    except Exception as e:
        print_yellow(f"Cannot register {func=} {first_param_type=}. {e=}")

