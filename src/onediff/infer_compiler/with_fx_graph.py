import os
import torch
import torch.fx as fx
import oneflow as flow
from torch.fx.node import map_aggregate
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from .obj_1f_from_torch import replace_obj, replace_func, replace_class, ProxySubmodule

def fx_node_tranform(gm):
    of_gm = to_of_transform(gm)

    enable_graph = os.getenv("with_graph", "False").lower() in (
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
                self.fx_md = of_gm
                self.config.enable_cudnn_conv_heuristic_search_algo(False)
                self.config.allow_fuse_add_to_output(True)
            
            def build(self, *args, **kwargs):
                return self.fx_md(*args, **kwargs)
        
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

    proxy_md = ProxySubmodule(origin_mod)
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
            self._modules[n] = _get_module(m, torch2flow)
        
        for k, v in proxy_md.__dict__.items():
            if k not in self.__dict__:
                try:
                    attr = getattr(proxy_md, k)
                except:
                    continue
                self.__dict__[k] = attr
    
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

def _get_of_module_list(origin_mod, torch2flow):
    assert isinstance(origin_mod, torch.nn.ModuleList)
    if origin_mod in torch2flow:
        return torch2flow[origin_mod]
    of_md_list = flow.nn.ModuleList()
    for m in origin_mod:
        of_md_list.append(_get_of_module(m, torch2flow))
    torch2flow[origin_mod] = of_md_list
    return of_md_list

def _get_of_module(origin_mod, torch2flow):
    if origin_mod in torch2flow:
        return torch2flow[origin_mod]

    if isinstance(origin_mod, torch.nn.ModuleList):
        return _get_of_module_list(origin_mod, torch2flow)

    proxy_md = ProxySubmodule(origin_mod)
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
            self._modules[n] = _get_of_module(m, torch2flow)
        
        for k, v in proxy_md.__dict__.items():
            if k not in self.__dict__:
                try:
                    attr = getattr(proxy_md, k)
                except:
                    continue
                self.__dict__[k] = attr
    
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


class UNetGraph(flow.nn.Graph):
    @flow.nn.Graph.with_dynamic_input_shape(size=9)
    def __init__(self, unet):
        super().__init__(enable_get_runtime_state_dict=True)
        self.unet = unet
        self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.config.allow_fuse_add_to_output(True)
        self.debug(0)

        os.environ["ONEFLOW_MLIR_CSE"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL"] = "1"
        # Open this will raise error
        # os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
        os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
        os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"

    def build(self, latent_model_input, t, encoder_hidden_states, cross_attention_kwargs, added_cond_kwargs=None, return_dict=False):
        encoder_hidden_states = flow._C.amp_white_identity(encoder_hidden_states)
        pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict,
        )
        return pred

    def warmup_with_load(self, file_path):
        state_dict = flow.load(file_path)
        self.load_runtime_state_dict(state_dict)

    def save_graph(self, file_path):
        state_dict = self.runtime_state_dict()
        flow.save(state_dict, file_path)
