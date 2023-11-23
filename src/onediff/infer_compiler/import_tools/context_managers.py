import sys
import torch
import oneflow as flow
from contextlib import contextmanager
from ..utils.log_utils import LOGGER


@contextmanager
def onediff_mock_torch(pkg_name=None):
    # Fixes  check the 'version'  error.
    attr_name = "__version__"
    restore_funcs = []  # Backup
    if hasattr(flow, attr_name) and hasattr(torch, attr_name):
        orig_flow_attr = getattr(flow, attr_name)
        restore_funcs.append(lambda: setattr(flow, attr_name, orig_flow_attr))
        setattr(flow, attr_name, getattr(torch, attr_name))

    backup_modules = sys.modules.copy()  # sys.modules has been changed
    # https://docs.oneflow.org/master/cookies/oneflow_torch.html
    with flow.mock_torch.enable(lazy=True):
        yield

    for restore_func in restore_funcs:
        restore_func()

    need_backup = len(sys.modules.copy()) != len(backup_modules)
    LOGGER.debug(f'PKG_NAME: {pkg_name} need_backup: {need_backup}')
    if need_backup:
        sys.modules = backup_modules
