import torch
from comfy import model_management
from comfy.model_base import SVD_img2vid
from onediff.infer_compiler import oneflow_compile
from register_comfy import DeepCacheUNet, FastDeepCacheUNet

from .booster_utils import set_environment_for_svd_img2vid

from .model_patcher import OneFlowDeepCacheSpeedUpModelPatcher


def deep_cache_speedup(
    model,
    use_graph,
    cache_interval,
    cache_layer_id,
    cache_block_id,
    start_step,
    end_step,
    *,
    gen_compile_options=None,
    use_oneflow_deepcache_speedup_modelpatcher=True,
):

    offload_device = model_management.unet_offload_device()
    if use_oneflow_deepcache_speedup_modelpatcher:
        model_patcher = OneFlowDeepCacheSpeedUpModelPatcher(
            model.model,
            load_device=model_management.get_torch_device(),
            offload_device=offload_device,
            cache_layer_id=cache_layer_id,
            cache_block_id=cache_block_id,
            use_graph=use_graph,
            gen_compile_options=gen_compile_options,
        )
    else:
        model_patcher = model
        model_patcher.deep_cache_unet = DeepCacheUNet(
            model_patcher.model.diffusion_model, cache_layer_id, cache_block_id
        )
        model_patcher.fast_deep_cache_unet = FastDeepCacheUNet(
            model_patcher.model.diffusion_model, cache_layer_id, cache_block_id
        )
        model_patcher.deep_cache_unet = oneflow_compile(model_patcher.deep_cache_unet)
        model_patcher.fast_deep_cache_unet = oneflow_compile(
            model_patcher.fast_deep_cache_unet
        )

    current_t = -1
    current_step = -1
    cache_h = None

    _first_run = True

    def apply_model(model_function, kwargs):
        if isinstance(model_patcher.model, SVD_img2vid):
            set_environment_for_svd_img2vid(model_patcher)
        nonlocal current_t, current_step, cache_h, _first_run

        if _first_run:
            if hasattr(model_patcher.deep_cache_unet, "quantize"):
                model_patcher.deep_cache_unet.quantize()

            if hasattr(model_patcher.fast_deep_cache_unet, "quantize"):
                model_patcher.fast_deep_cache_unet.quantize()
            _first_run = False

        xa = kwargs["input"]
        t = kwargs["timestep"]
        c_concat = kwargs["c"].get("c_concat", None)
        c_crossattn = kwargs["c"].get("c_crossattn", None)
        y = kwargs["c"].get("y", None)
        control = kwargs["c"].get("control", None)
        transformer_options = kwargs["c"].get("transformer_options", None)

        # https://github.com/comfyanonymous/ComfyUI/blob/629e4c552cc30a75d2756cbff8095640af3af163/comfy/model_base.py#L51-L69
        sigma = t
        xc = model_patcher.model.model_sampling.calculate_input(sigma, xa)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = model_patcher.model.get_dtype()
        xc = xc.to(dtype)
        t = model_patcher.model.model_sampling.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "to"):
                extra = extra.to(dtype)
            extra_conds[o] = extra

        x = xc
        timesteps = t
        y = None if y is None else y.to(dtype)
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["current_index"] = 0
        transformer_patches = transformer_options.get("patches", {})
        """
            Apply the model to an input batch.
            :param x: an [N x C x ...] Tensor of inputs.
            :param timesteps: a 1-D batch of timesteps.
            :param context: conditioning plugged in via crossattn
            :param y: an [N] Tensor of labels, if class-conditional.
            :return: an [N x C x ...] Tensor of outputs.
            """

        # reference https://gist.github.com/laksjdjf/435c512bc19636e9c9af4ee7bea9eb86
        if t[0].item() > current_t:
            current_step = -1

        current_t = t[0].item()
        apply = 1000 - end_step <= current_t <= 1000 - start_step  # t is 999->0

        if apply:
            current_step += 1
        else:
            current_step = -1
        current_t = t[0].item()

        is_slow_step = current_step % cache_interval == 0 and apply

        model_output = None
        if is_slow_step:
            cache_h = None
            model_output, cache_h = model_patcher.deep_cache_unet(
                x,
                timesteps,
                context,
                y,
                control,
                transformer_options,
                **extra_conds,
            )
        else:
            model_output, cache_h = model_patcher.fast_deep_cache_unet(
                x,
                cache_h,
                timesteps,
                context,
                y,
                control,
                transformer_options,
                **extra_conds,
            )

        return model_patcher.model.model_sampling.calculate_denoised(
            sigma, model_output, xa
        )

    model_patcher.set_model_unet_function_wrapper(apply_model)
    return (model_patcher,)
