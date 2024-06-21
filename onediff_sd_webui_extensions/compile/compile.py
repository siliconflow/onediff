from modules.sd_hijack import apply_optimizations

from onediff.infer_compiler import compile, oneflow_compile

from .utils import OneDiffCompiledGraph, disable_unet_checkpointing
from .quantization import quant_unet_oneflow


def get_compiled_graph(
    sd_model, *, backend, quantization=None, options=None
) -> OneDiffCompiledGraph:
    diffusion_model = sd_model.model.diffusion_model
    compiled_unet = onediff_compile(
        diffusion_model, backend=backend, quantization=quantization, options=options
    )
    return OneDiffCompiledGraph(sd_model, compiled_unet, quantization)


def onediff_compile(unet_model, *, quantization=False, backend="oneflow", options=None):
    if backend == "oneflow":
        return compile_unet_oneflow(
            unet_model, quantization=quantization, options=options
        )
    elif backend == "nexfort":
        return compile_unet_nexfort(
            unet_model, quantization=quantization, options=options
        )
    else:
        raise NotImplementedError(f"Can't find backend {backend} for OneDiff")


def compile_unet_oneflow(unet_model, *, quantization=False, options=None):
    from .oneflow.utils import init_oneflow_backend

    # 1. register mock map for converting torch to oneflow
    init_oneflow_backend()

    # 2. (for controlnet) remove attr forward to prevent mock failing
    if "forward" in unet_model.__dict__:
        unet_model.__dict__.pop("forward")

    # 3. disable checkpoint to prevent mock failing
    disable_unet_checkpointing(unet_model)

    compiled_unet_model = oneflow_compile(unet_model, options=options)
    if quantization:
        compiled_unet_model = quant_unet_oneflow(compiled_unet_model)
    return compiled_unet_model


def compile_unet_nexfort(unet_model, *, quantization=False, options=None):
    if quantization:
        raise NotImplementedError(
            "Quantization for nexfort backend is not implemented yet."
        )
    from .nexfort.utils import init_nexfort_backend

    init_nexfort_backend()
    apply_optimizations("nexfort")
    disable_unet_checkpointing(unet_model)
    unet_model.convert_to_fp16()
    return compile(unet_model, backend="nexfort", options=options)
