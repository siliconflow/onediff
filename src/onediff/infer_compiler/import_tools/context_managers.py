import sys
import torch
import oneflow as flow
from contextlib import contextmanager
from ..utils.log_utils import LOGGER

_PACKAGE_MAP = {}
def cache_package(key, package):
    global _PACKAGE_MAP
    _PACKAGE_MAP[key] = package


@contextmanager
def onediff_mock_torch(pkg_name=None):
    # Fixes  check the 'version'  error.
    attr_name = "__version__"
    restore_funcs = []  # Backup
    if hasattr(flow, attr_name) and hasattr(torch, attr_name):
        orig_flow_attr = getattr(flow, attr_name)
        restore_funcs.append(lambda: setattr(flow, attr_name, orig_flow_attr))
        setattr(flow, attr_name, getattr(torch, attr_name))
    
    backup = sys.modules.copy()
    # https://docs.oneflow.org/master/cookies/oneflow_torch.html
    pkg = _PACKAGE_MAP.get(pkg_name, None)
    if pkg is None:
        with flow.mock_torch.enable(lazy=True):
            yield pkg
    else:
        yield pkg

    for restore_func in restore_funcs:
        restore_func()
        
    # https://docs.python.org/3/library/sys.html?highlight=sys%20modules#sys.modules
    need_backup = len(sys.modules.copy()) != len(backup)
    if need_backup:
        sys.modules = backup
