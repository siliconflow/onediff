import sys
import torch
import oneflow as flow
from contextlib import contextmanager

_BACKUP_MODULES = {} # 为mock后的pkg包 各自保存了一份值为模块的字典
def add_backup_modules(key, module):
    _BACKUP_MODULES[key] = module
    

@contextmanager
def onediff_mock_torch(pkg_name=None):
    # Fixes  check the 'version'  error.
    attr_name = "__version__"
    restore_funcs = []  # Backup
    if hasattr(flow, attr_name) and hasattr(torch, attr_name):
        orig_flow_attr = getattr(flow, attr_name)
        restore_funcs.append(lambda: setattr(flow, attr_name, orig_flow_attr))
        setattr(flow, attr_name, getattr(torch, attr_name))

    backup_modules = sys.modules.copy() # sys.modules has been changed
    if pkg_name is not None and pkg_name in _BACKUP_MODULES:
        print(f'pkg_name: {pkg_name}')
        package = _BACKUP_MODULES[pkg_name]
    else:
        package = None
    # https://docs.oneflow.org/master/cookies/oneflow_torch.html
    with flow.mock_torch.enable(lazy=True):
        yield package

    for restore_func in restore_funcs:
        restore_func()

    def diff_modules(a, b):
        return set(a.keys()) - set(b.keys())
    def diff_values(a, b):
        return set(a.values()) - set(b.values())
    x = diff_modules(sys.modules, backup_modules)

    print(f'x: {x}')
    
    y = diff_values(sys.modules, backup_modules)
    print(f'y: {y}')


    # sys.modules = backup_modules
    

