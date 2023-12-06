import torch
import oneflow as flow
from oneflow.framework.args_tree import ArgsTree

from onediff.infer_compiler.transform import proxy_class

from oneflow.framework.args_tree import _ONEFLOW_ARGS_TREE_CUSTOM_TYPE_DICT
import types

import diffusers
AutoencoderKLOutputOflow = proxy_class(diffusers.models.modeling_outputs.AutoencoderKLOutput)
def flattened_iter(self):
    latent_dist = self.latent_dist
    flattened = [latent_dist.parameters,
                 latent_dist.mean,
                 latent_dist.logvar,
                 latent_dist.deterministic,
                 latent_dist.std,
                 latent_dist.var,
                 ]
    return iter(flattened)

def flatten_cons(self, *args):
    self_cls = self.__class__
    self_latent_dist_cls = self.latent_dist.__class__
    latent_dist = self_latent_dist_cls.__new__(self_latent_dist_cls)
    latent_dist.parameters = args[0]
    latent_dist.mean = args[1]
    latent_dist.logvar = args[2]
    latent_dist.deterministic = args[3]
    latent_dist.std = args[4]
    latent_dist.var = args[5]
    return self_cls(latent_dist=latent_dist)

def GetAutoencoderKLOutputOflowIter(self):
    return AutoencoderKLOutputOflowIter(self)
AutoencoderKLOutputOflow._flatten_iter = flattened_iter
AutoencoderKLOutputOflow._flatten_cons= flatten_cons

_ONEFLOW_ARGS_TREE_CUSTOM_TYPE_DICT[AutoencoderKLOutputOflow] = "yes"

def transform_arg(torch_obj):
    if isinstance(torch_obj, proxy_class(diffusers.models.modeling_outputs.AutoencoderKLOutput)):
        obj = torch_obj.latent_dist
        DiagonalGaussianDistribution = diffusers.models.vae.DiagonalGaussianDistribution
        dist = DiagonalGaussianDistribution.__new__(DiagonalGaussianDistribution)
        dist.parameters = flow.utils.tensor.to_torch(obj.parameters)
        dist.mean = flow.utils.tensor.to_torch(obj.mean)
        dist.logvar = flow.utils.tensor.to_torch(obj.logvar)
        dist.deterministic = obj.deterministic
        dist.std = flow.utils.tensor.to_torch(obj.std)
        dist.var = flow.utils.tensor.to_torch(obj.var)

        return diffusers.models.modeling_outputs.AutoencoderKLOutput(latent_dist=dist)
    else:
        return torch_obj


def input_output_processor(func):
    def process_input(*args, **kwargs):
        def input_fn(value):
            if isinstance(value, torch.Tensor):
                # TODO: https://github.com/siliconflow/sd-team/issues/109
                return flow.utils.tensor.from_torch(value.contiguous())
            else:
                return value

        args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
        out = args_tree.map_leaf(input_fn)
        mapped_args = out[0]
        mapped_kwargs = out[1]
        return mapped_args, mapped_kwargs

    def process_output(output):
        def output_fn(value):
            if isinstance(value, flow.Tensor):
                return flow.utils.tensor.to_torch(value)
            else:
                return transform_arg(value)

        out_tree = ArgsTree((output, None), False)
        out = out_tree.map_leaf(output_fn)
        return out[0]

    def wrapper(cls, *args, **kwargs):
        mapped_args, mapped_kwargs = process_input(*args, **kwargs)
        output = func(cls, *mapped_args, **mapped_kwargs)
        return process_output(output)

    return wrapper
