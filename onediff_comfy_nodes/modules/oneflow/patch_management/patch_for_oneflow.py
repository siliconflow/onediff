"""oneflow/framework/args_tree.py
fix: TypeError: can only concatenate str (not "tuple") to str
TODO: fix in oneflow
"""
import oneflow as flow
from oneflow.framework.args_tree import NamedArg


class PatchNamedArg(NamedArg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        name = getattr(self, "_name", None)
        setattr(self, "_name", str(name) if name is not None else None)


flow.framework.args_tree.NamedArg = PatchNamedArg



original_copy_ = flow.Tensor.copy_

def new_copy_(self, src, *args, **kwargs):
    # print(f'{__file__}.new_copy_ {self.dtype=}')
    if self.dtype == flow.int8 and src.dtype != flow.int8:
        return
    return original_copy_(self, *args, **kwargs)

# Replace the original copy_ method with the new one
flow.Tensor.copy_ = new_copy_
