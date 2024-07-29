# /ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/sampling.py
import oneflow as flow  # usort: skip
from einops import rearrange
from onediff.infer_compiler import DeployableModule
from onediff.infer_compiler.backends.oneflow.transform import register
from oneflow.nn.functional import group_norm

from ._config import animatediff_hijacker, animatediff_of, animatediff_pt, comfy_of

FunctionInjectionHolder = animatediff_pt.animatediff.sampling.FunctionInjectionHolder


def cast_bias_weight(s, input):
    bias = None
    # non_blocking = comfy.model_management.device_supports_non_blocking(input.device)
    non_blocking = False
    if s.bias is not None:
        bias = s.bias.to(
            device=input.device, dtype=input.dtype, non_blocking=non_blocking
        )
    weight = s.weight.to(
        device=input.device, dtype=input.dtype, non_blocking=non_blocking
    )
    return weight, bias


def groupnorm_mm_factory(params, manual_cast=False):
    def groupnorm_mm_forward(self, input):
        # axes_factor normalizes batch based on total conds and unconds passed in batch;
        # the conds and unconds per batch can change based on VRAM optimizations that may kick in
        if not params.is_using_sliding_context():
            batched_conds = input.size(0) // params.full_length
        else:
            batched_conds = input.size(0) // params.context_options.context_length

        # input = rearrange(input, "(b f) c h w -> b c f h w", b=batched_conds)
        input = input.unflatten(0, (batched_conds, -1)).permute(0, 2, 1, 3, 4)

        if manual_cast:
            # weight, bias = comfy_of.ops.cast_bias_weight(self, input)
            weight, bias = cast_bias_weight(self, input)
        else:
            weight, bias = self.weight, self.bias
        input = group_norm(input, self.num_groups, weight, bias, self.eps)
        # input = rearrange(input, "b c f h w -> (b f) c h w", b=batched_conds)
        input = input.permute(0, 2, 1, 3, 4).flatten(0, 1)
        return input

    return groupnorm_mm_forward


# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/motion_module_ad.py
AnimateDiffVersion = animatediff_pt.animatediff.motion_module_ad.AnimateDiffVersion
AnimateDiffFormat = animatediff_pt.animatediff.motion_module_ad.AnimateDiffFormat
# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/utils_model.py
ModelTypeSD = animatediff_pt.animatediff.utils_model.ModelTypeSD


class Handles:
    def __init__(self):
        self.handles = []

    def add(self, obj, key, value):
        org_attr = getattr(obj, key, None)
        setattr(obj, key, value)
        self.handles.append(lambda: setattr(obj, key, org_attr))

    def restore(self):
        for handle in self.handles:
            handle()
        self.handles = []


handles = Handles()


def inject_functions(orig_func, self, model, params):

    ret = orig_func(self, model, params)

    if model.motion_models is not None:
        # only apply groupnorm hack if not [v3 or ([not Hotshot] and SD1.5 and v2 and apply_v2_properly)]
        info = model.motion_models[0].model.mm_info
        if not (
            info.mm_version == AnimateDiffVersion.V3
            or (
                info.mm_format not in [AnimateDiffFormat.HOTSHOTXL]
                and info.sd_type == ModelTypeSD.SD1_5
                and info.mm_version == AnimateDiffVersion.V2
                and params.apply_v2_properly
            )
        ):

            handles.add(flow.nn.GroupNorm, "forward", groupnorm_mm_factory(params))
            # comfy_of.ops.manual_cast.GroupNorm.forward_comfy_cast_weights = groupnorm_mm_factory(params, manual_cast=True)
            handles.add(
                comfy_of.ops.manual_cast.GroupNorm,
                "forward_comfy_cast_weights",
                groupnorm_mm_factory(params, manual_cast=True),
            )

        del info
    return ret


def restore_functions(orig_func, *args, **kwargs):
    ret = orig_func(*args, **kwargs)
    handles.restore()
    return ret


def cond_func(orig_func, self, model, *args, **kwargs):
    diff_model = model.model.diffusion_model
    if isinstance(diff_model, DeployableModule):
        return True
    else:
        return False


animatediff_hijacker.register(
    FunctionInjectionHolder.inject_functions,
    inject_functions,
    cond_func,
)

animatediff_hijacker.register(
    FunctionInjectionHolder.restore_functions,
    restore_functions,
    cond_func,
)
