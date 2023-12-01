import sys
import torch
import oneflow as flow
from contextlib import contextmanager


@contextmanager
def onediff_mock_torch():
    # Fixes  check the 'version'  error.
    attr_name = "__version__"
    restore_funcs = []  # Backup
    if hasattr(flow, attr_name) and hasattr(torch, attr_name):
        orig_flow_attr = getattr(flow, attr_name)
        restore_funcs.append(lambda: setattr(flow, attr_name, orig_flow_attr))
        setattr(flow, attr_name, getattr(torch, attr_name))

    # https://docs.oneflow.org/master/cookies/oneflow_torch.html
    with flow.mock_torch.enable(lazy=True):
        yield

    for restore_func in restore_funcs:
        restore_func()
