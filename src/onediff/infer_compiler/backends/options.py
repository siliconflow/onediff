from typing import Dict


class CompileOptions:
    def __init__(self, dynamic=True, oneflow=None, nexfort=None):
        from .oneflow import OneflowCompileOptions

        self.dynamic = dynamic
        self.oneflow = oneflow if oneflow is not None else OneflowCompileOptions()
        self.nexfort = nexfort if nexfort is not None else dict()


# a global default compile options
_GLOBAL_compile_options = CompileOptions()
