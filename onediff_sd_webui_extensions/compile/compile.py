from modules.sd_hijack import apply_optimizations

from onediff.infer_compiler import compile, oneflow_compile

from .utils import OneDiffCompiledGraph, disable_unet_checkpointing


def get_compiled_graph(sd_model, quantization, *, options=None) -> OneDiffCompiledGraph:
    diffusion_model = sd_model.model.diffusion_model
    # TODO: quantization
    if quantization is True:
        raise
    compiled_unet = onediff_compile(diffusion_model, options=options)
    return OneDiffCompiledGraph(sd_model, compiled_unet, quantization)


def onediff_compile(unet_model, *, backend="oneflow", options=None):
    if backend == "oneflow":
        # for controlnet
        if "forward" in unet_model.__dict__:
            unet_model.__dict__.pop("forward")
        return compile_unet_oneflow(unet_model, options=options)
    elif backend == "nexfort":
        return compile_unet_nexfort(unet_model, options=options)
    else:
        raise NotImplementedError(f"Can't find backend {backend} for OneDiff")


def compile_unet_oneflow(unet_model, *, options=None):
    from .oneflow.utils import init_oneflow_backend

    init_oneflow_backend()
    disable_unet_checkpointing(unet_model)
    return oneflow_compile(unet_model, options=options)


def compile_unet_nexfort(unet_model, *, options=None):
    from .nexfort.utils import init_nexfort_backend

    init_nexfort_backend()
    apply_optimizations("nexfort")
    disable_unet_checkpointing(unet_model)
    unet_model.convert_to_fp16()
    return compile(unet_model, backend="nexfort", options=options)
