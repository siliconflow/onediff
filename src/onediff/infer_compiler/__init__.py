import os
from collections import OrderedDict
import torch
import torch.fx as fx
import diffusers
import oneflow
import oneflow as flow
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import map_aggregate
from torch.func import functionalize
import importlib
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from .attention_1f import BasicTransformerBlock, FeedForward, GEGLU
from .attention_processor_1f import Attention
from .lora_1f import LoRACompatibleLinear


def replace_class(cls):
    if cls.__module__.startswith("torch"):
        mod_name = cls.__module__.replace("torch", "oneflow")
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls.__name__)
    elif cls == diffusers.models.attention.BasicTransformerBlock:
        return BasicTransformerBlock
    elif cls == diffusers.models.attention_processor.Attention:
        return Attention
    elif cls == diffusers.models.attention.FeedForward:
        return FeedForward
    elif cls == diffusers.models.attention.GEGLU:
        return GEGLU
    elif cls == diffusers.models.lora.LoRACompatibleLinear:
        return LoRACompatibleLinear


def replace_obj(obj):
    cls = type(obj)
    if cls == torch.dtype:
        return {
            "torch.float16": flow.float16,
            "torch.float32": flow.float32,
            "torch.double": flow.double,
            "torch.int8": flow.int8,
            "torch.int32": flow.int32,
            "torch.int64": flow.int64,
            "torch.uint8": flow.uint8,
        }[str(obj)]
    if cls == torch.fx.immutable_collections.immutable_list:
        return [e for e in obj]
    replacement = replace_class(cls)
    if replacement is not None:
        if cls in [torch.device]:
            return replacement(str(obj))
        elif cls == torch.nn.parameter.Parameter:
            return flow.utils.tensor.from_torch(obj.data)
        else:
            raise RuntimeError("don't know how to create oneflow obj for: " + str(cls))
    else:
        return obj


def replace_func(func):
    if func == torch.conv2d:
        return oneflow.nn.functional.conv2d
    if func == torch._C._nn.linear:
        return oneflow.nn.functional.linear
    if func.__module__.startswith("torch"):
        mod_name = func.__module__.replace("torch", "oneflow")
        mod = importlib.import_module(mod_name)
        return getattr(mod, func.__name__)
    else:
        return func


def map_args(args, kwargs):
    args = [replace_obj(a) for a in args]
    kwargs = dict((k, replace_obj(v)) for (k, v) in kwargs.items())
    return (args, kwargs)


class ProxySubmodule:
    def __init__(self, submod):
        self._1f_proxy_submod = submod
        self._1f_proxy_parameters = dict()
        self._1f_proxy_children = dict()

    def __getattribute__(self, attribute):
        if attribute.startswith("_1f_proxy"):
            return object.__getattribute__(self, attribute)
        elif attribute in ["forward", "_conv_forward"]:
            replacement = replace_class(type(self._1f_proxy_submod))
            return lambda *args, **kwargs: getattr(replacement, attribute)(
                self, *args, **kwargs
            )
        elif (
            isinstance(
                self._1f_proxy_submod, diffusers.models.attention_processor.Attention
            )
            and attribute == "get_attention_scores"
        ):
            replacement = replace_class(type(self._1f_proxy_submod))
            return lambda *args, **kwargs: getattr(replacement, attribute)(
                self, *args, **kwargs
            )
        elif (
            isinstance(self._1f_proxy_submod, torch.nn.Linear)
            and attribute == "use_fused_matmul_bias"
        ):
            return (
                self.bias is not None
                and os.getenv("ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR") == "1"
            )
        elif (
            isinstance(self._1f_proxy_submod, torch.nn.Dropout)
            and attribute == "generator"
        ):
            return flow.Generator()
        elif (
            isinstance(self._1f_proxy_submod, torch.nn.Conv2d)
            and attribute == "channel_pos"
        ):
            return "channels_first"
        else:
            a = getattr(self._1f_proxy_submod, attribute)
            if isinstance(a, torch.nn.parameter.Parameter):
                # TODO(oneflow): assert a.requires_grad == False
                if attribute not in self._1f_proxy_parameters:
                    a = flow.utils.tensor.from_torch(a.data)
                    self._1f_proxy_parameters[attribute] = a
                else:
                    a = self._1f_proxy_parameters[attribute]
            elif isinstance(a, torch.nn.ModuleList):
                a = [ProxySubmodule(m) for m in a]
            elif isinstance(a, torch.nn.Module):
                if attribute not in self._1f_proxy_children:
                    a = ProxySubmodule(a)
                    self._1f_proxy_children[attribute] = a
                else:
                    a = self._1f_proxy_children[attribute]
            assert (
                type(a).__module__.startswith("torch") == False
                and type(a).__module__.startswith("diffusers") == False
            ), "can't be a torch module at this point! But found " + str(type(a))
            return a

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        replacement = replace_class(type(self._1f_proxy_submod))
        if replacement is not None:
            return replacement.__call__(self, *args, **kwargs)
        else:
            raise RuntimeError(
                "can't find oneflow module for: " + str(type(self._1f_proxy_submod))
            )


class OneFlowInterpreter(torch.fx.Interpreter):
    from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate

    def call_function(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
        args, kwargs = map_args(args, kwargs)
        target = replace_func(target)
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


def torchbackend(gm, example_inputs):
    with_interp = os.getenv("with_interp", "True").lower() in (
        "true",
        "1",
        "t",
    )
    print("spt========== beforetrans")
    def wrapped_forward(*args, **kwargs):
        args = [flow.utils.tensor.from_torch(a) for a in args]
        if with_interp:
            output = OneFlowInterpreter(gm, garbage_collect_values=False).run(
                *args, **kwargs
            )
        else:
            transformed_fn = fx_node_tranform(gm)
            import pdb; pdb.set_trace()
            output = transformed_fn(*args, **kwargs)
        if isinstance(output, tuple):
            return tuple(flow.utils.tensor.to_torch(i) for i in output)
        return flow.utils.tensor.to_torch(output)

    return wrapped_forward

def fx_node_tranform(gm):
    of_gm = to_of_transform(gm)
    import pdb; pdb.set_trace()

    enable_graph = os.getenv("enable_graph", "False").lower() in (
        "true",
        "1",
        "t",
    )

    if not enable_graph:
        oneflow_fn = of_gm.forward
    else:
        class OfGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = of_gm
            
            def build(self, *args, **kwargs):
                return self.m(*args, **kwargs)
        
        of_g = OfGraph()
        of_g.debug(0)
        oneflow_fn = lambda *args, **kwargs: of_g(*args, **kwargs)

    return oneflow_fn


def to_of_transform(
    gm: torch.fx.GraphModule, tracer_class: type = fx.Tracer
) -> torch.fx.GraphModule:
    name2node = {}
    name2obj = {}
    torch2flow = {}
    of_g = flow.fx.Graph()
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        print("old: ", node.format_node())
        if node.op == "placeholder":
            of_node = of_g.create_node('placeholder', node.target)
            name2node[node.name] = of_node
        elif node.op == "output":
            of_node = of_g.output(node_replace_args(node.args, name2node)[0])
            name2node[node.name] = of_node
        elif node.op == "call_function":
            of_node = of_g.create_node('call_function', replace_func(node.target), args=node_replace_args(node.args, name2node), kwargs=node_replace_args(node.kwargs, name2node))
            name2node[node.name] = of_node
        elif node.op == "call_method":
            of_node = of_g.create_node('call_method', node.target, args=node_replace_args(node.args, name2node), kwargs=node_replace_args(node.kwargs, name2node))
            name2node[node.name] = of_node
        elif node.op == "call_module":
            torch_md = modules[node.target]
            name2obj[node.target] = _get_module(torch_md, torch2flow)
            of_node = of_g.create_node('call_module', node.target, args=node_replace_args(node.args, name2node), kwargs=node_replace_args(node.kwargs, name2node))
            name2node[node.name] = of_node
        elif node.op == "get_attr":
            of_node = of_g.create_node('get_attr', node.target)
            name2node[node.name] = of_node
            name2obj[node.target] = _get_attr(gm, node, torch2flow)
        else:
            raise ValueError(f"not valid node type{node.foramt_node()}")
        print("new: ", of_node.format_node())

    print("\n new of graph", of_g.print_tabular())
    for of_node in of_g.nodes:
        print(of_node.format_node())
    
    of_gm = flow.fx.GraphModule(name2obj, of_g)
    of_gm.graph.lint()
    of_gm.recompile()
    return of_gm

def replace_node(node, name2node):
    if isinstance(node, torch.fx.Node):
        return name2node[node.name]
    else:
        return replace_obj(node)


def node_replace_args(args, name2node):
    return map_aggregate(args, lambda node: replace_node(node, name2node))

class NodeProxySubmodule:
    def __init__(self, submod):
        self._1f_proxy_submod = submod
        self._1f_proxy_parameters = dict()
        self._1f_proxy_children = dict()

    def __getattribute__(self, attribute):
        if attribute.startswith("_1f_proxy"):
            return object.__getattribute__(self, attribute)
        elif attribute in ["forward", "_conv_forward"]:
            replacement = replace_class(type(self._1f_proxy_submod))
            return lambda *args, **kwargs: getattr(replacement, attribute)(
                self, *args, **kwargs
            )
        elif (
            isinstance(
                self._1f_proxy_submod, diffusers.models.attention_processor.Attention
            )
            and attribute == "get_attention_scores"
        ):
            replacement = replace_class(type(self._1f_proxy_submod))
            return lambda *args, **kwargs: getattr(replacement, attribute)(
                self, *args, **kwargs
            )
        elif (
            isinstance(self._1f_proxy_submod, torch.nn.Linear)
            and attribute == "use_fused_matmul_bias"
        ):
            return (
                self.bias is not None
                and os.getenv("ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR") == "1"
            )
        elif (
            isinstance(self._1f_proxy_submod, torch.nn.Dropout)
            and attribute == "generator"
        ):
            return flow.Generator()
        elif (
            isinstance(self._1f_proxy_submod, torch.nn.Conv2d)
            and attribute == "channel_pos"
        ):
            return "channels_first"
        else:
            a = getattr(self._1f_proxy_submod, attribute)
            if isinstance(a, torch.nn.parameter.Parameter):
                # TODO(oneflow): assert a.requires_grad == False
                if attribute not in self._1f_proxy_parameters:
                    a = flow.utils.tensor.from_torch(a.data)
                    self._1f_proxy_parameters[attribute] = a
                else:
                    a = self._1f_proxy_parameters[attribute]
            elif isinstance(a, torch.nn.ModuleList):
                a = [NodeProxySubmodule(m) for m in a]
            elif isinstance(a, torch.nn.Module):
                if attribute not in self._1f_proxy_children:
                    a = NodeProxySubmodule(a)
                    self._1f_proxy_children[attribute] = a
                else:
                    a = self._1f_proxy_children[attribute]
            assert (
                type(a).__module__.startswith("torch") == False
                and type(a).__module__.startswith("diffusers") == False
            ), "can't be a torch module at this point! But found " + str(type(a))
            return a

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        replacement = replace_class(type(self._1f_proxy_submod))
        if replacement is not None:
            return replacement.__call__(self, *args, **kwargs)
        else:
            raise RuntimeError(
                "can't find oneflow module for: " + str(type(self._1f_proxy_submod))
            )

def _get_module_list(origin_mod, torch2flow):
    assert isinstance(origin_mod, torch.nn.ModuleList)
    if origin_mod in torch2flow:
        return torch2flow[origin_mod]
    of_md_list = flow.nn.ModuleList()
    for m in origin_mod:
        of_md_list.append(_get_module(m, torch2flow))
    torch2flow[origin_mod] = of_md_list
    return of_md_list


def _get_module(origin_mod, torch2flow):
    if origin_mod in torch2flow:
        return torch2flow[origin_mod]

    if isinstance(origin_mod, torch.nn.ModuleList):
        return _get_module_list(origin_mod, torch2flow)

    proxy_md = NodeProxySubmodule(origin_mod)
    new_md_cls = replace_class(type(origin_mod))

    def init(self):
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        for (n, p) in list(proxy_md.named_parameters("", False)):
            self._parameters[n] = flow.utils.tensor.from_torch(p.data)
        for (n, b) in list(proxy_md.named_buffers("", False)):
            self._buffers[n] = flow.utils.tensor.from_torch(b.data)
        for (n, m) in proxy_md._modules.items():
            # TODO
            self._modules[n] = _get_module(m, torch2flow)
    
    def proxy_getattr(self, attr):
        if attr in ["_parameters", "_buffers", "_modules"]:
            raise ValueError(f"missing attr {attr} in base class")
        else:
            return getattr(proxy_md, attr)

    of_md_cls = type(
        str(new_md_cls), (new_md_cls,), {"__init__": init, "__getattr__": proxy_getattr},
    )

    new_md = of_md_cls()

    torch2flow[origin_mod] = new_md
    return new_md

def _get_attr(gm, node, torch2flow):
    attr = getattr(gm, node.target)
    if attr in torch2flow:
        return torch2flow[attr]
    of_attr = replace_obj(attr)
    torch2flow[attr] = of_attr
    return of_attr
