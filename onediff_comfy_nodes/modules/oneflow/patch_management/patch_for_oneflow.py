"""oneflow/framework/args_tree.py
fix: TypeError: can only concatenate str (not "tuple") to str
TODO: fix in oneflow
"""
import oneflow as flow  # usort: skip
from oneflow.framework.args_tree import NamedArg


class PatchNamedArg(NamedArg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        name = getattr(self, "_name", None)
        setattr(self, "_name", str(name) if name is not None else None)


flow.framework.args_tree.NamedArg = PatchNamedArg
