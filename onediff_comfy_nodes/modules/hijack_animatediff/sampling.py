# /home/fengwen/quant/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/sampling.py
from einops import rearrange
from oneflow.nn.functional import group_norm
import oneflow as flow
from ._config import animatediff_pt, animatediff_hijacker, animatediff_of

FunctionInjectionHolder = animatediff_pt.animatediff.sampling.FunctionInjectionHolder
ADGS = animatediff_pt.animatediff.sampling.ADGS


def groupnorm_mm_factory(params):
    def groupnorm_mm_forward(self, input):
        # axes_factor normalizes batch based on total conds and unconds passed in batch;
        # the conds and unconds per batch can change based on VRAM optimizations that may kick in
        if not ADGS.is_using_sliding_context():
            axes_factor = input.size(0) // params.video_length
        else:
            axes_factor = input.size(0) // params.context_length
        input = rearrange(input, "(b f) c h w -> b c f h w", b=axes_factor)
        input = group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        input = rearrange(input, "b c f h w -> (b f) c h w", b=axes_factor)
        return input

    return groupnorm_mm_forward


# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/motion_utils.py
GroupNormAD_OF_CLS = animatediff_of.animatediff.motion_utils.GroupNormAD
# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/motion_module_ad.py
AnimateDiffVersion = animatediff_pt.animatediff.motion_module_ad.AnimateDiffVersion
AnimateDiffFormat = animatediff_pt.animatediff.motion_module_ad.AnimateDiffFormat
# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/model_utils.py
ModelTypeSD = animatediff_pt.animatediff.model_utils.ModelTypeSD

def inject_functions(orig_func, self, model, params):
    ret = orig_func(self, model, params)
    # TODO support check info
    info = model.motion_model.model.mm_info
    if not (info.mm_version == AnimateDiffVersion.V3 or (info.mm_format == AnimateDiffFormat.ANIMATEDIFF and info.sd_type == ModelTypeSD.SD1_5 and
            info.mm_version == AnimateDiffVersion.V2 and params.apply_v2_models_properly)):
        flow.nn.GroupNorm.forward = groupnorm_mm_factory(params)
        if params.apply_mm_groupnorm_hack:
            GroupNormAD_OF_CLS.forward = groupnorm_mm_factory(params)

    return ret


# TODO support restore


def cond_func(*args, **kwargs):
    return True


animatediff_hijacker.register(
    FunctionInjectionHolder.inject_functions,
    inject_functions,
    cond_func,
)
