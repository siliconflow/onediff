import torch
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from .transform import map_args, ProxySubmodule


class OneFlowInterpreter(torch.fx.Interpreter):
    from torch.fx.node import Argument, Target

    def call_function(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        args, kwargs = map_args(args, kwargs)
        target = torch2oflow(target)
        return super().call_function(target, args, kwargs)

    def call_method(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        args, kwargs = map_args(args, kwargs)
        return super().call_method(target, args, kwargs)

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        submod = self.fetch_attr(target)
        submod = ProxySubmodule(submod)
        return submod(*args, **kwargs)
