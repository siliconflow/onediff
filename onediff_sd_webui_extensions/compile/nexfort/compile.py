from modules.sd_hijack import apply_optimizations
from onediff_utils import disable_unet_checkpointing

from onediff.infer_compiler import compile

from .utils import init_nexfort


def nexfort_compile_ldm_unet(unet_model, *, options=None):
    init_nexfort()
    apply_optimizations("nexfort")
    disable_unet_checkpointing(unet_model)
    unet_model.convert_to_fp16()
    return compile(unet_model, backend="nexfort", options=options)
