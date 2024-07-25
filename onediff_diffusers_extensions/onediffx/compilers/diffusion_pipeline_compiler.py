import functools
import json
import os

import torch
from onediff.infer_compiler import compile, DeployableModule
from onediff.utils import logger


def _recursive_getattr(obj, attr, default=None):
    attrs = attr.split(".")
    for attr in attrs:
        if not hasattr(obj, attr):
            return default
        obj = getattr(obj, attr, default)
    return obj


def _recursive_setattr(obj, attr, value):
    attrs = attr.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


_PARTS = [
    "text_encoder",
    "text_encoder_2",
    "image_encoder",
    "unet",
    "controlnet",
    "fast_unet",  # for deepcache
    "prior",  # for StableCascadePriorPipeline
    "decoder",  # for StableCascadeDecoderPipeline
    "transformer",  # for Transformer-based DiffusionPipeline such as DiTPipeline and PixArtAlphaPipeline
    "vqgan.down_blocks",  # for StableCascadeDecoderPipeline
    "vqgan.up_blocks",  # for StableCascadeDecoderPipeline
    "vae.decoder",
    "vae.encoder",
]


def _filter_parts(ignores=()):
    filtered_parts = []
    for part in _PARTS:
        skip = False
        for ignore in ignores:
            if part == ignore or part.startswith(ignore + "."):
                skip = True
                break
        if not skip:
            filtered_parts.append(part)

    return filtered_parts


def compile_pipe(
    pipe,
    *,
    backend="oneflow",
    options=None,
    ignores=(),
    fuse_qkv_projections=False,
):
    if fuse_qkv_projections:
        pipe = fuse_qkv_projections_in_pipe(pipe)

    if backend == "nexfort" and isinstance(options, str):
        import json

        options = json.loads(options)

    if backend == "nexfort" and options is not None and "memory_format" in options:
        memory_format = getattr(torch, options["memory_format"])
        pipe = convert_pipe_to_memory_format(
            pipe, ignores=ignores, memory_format=memory_format
        )
        del options["memory_format"]

    # To fix the bug of graph load of vae. Please refer to: https://github.com/siliconflow/onediff/issues/452
    if (
        hasattr(pipe, "upcast_vae")
        and pipe.vae.dtype == torch.float16
        and pipe.vae.config.force_upcast
    ):
        pipe.upcast_vae()

    filtered_parts = _filter_parts(ignores=ignores)
    for part in filtered_parts:
        obj = _recursive_getattr(pipe, part, None)
        if obj is not None:
            logger.info(f"Compiling {part}")
            _recursive_setattr(
                pipe, part, compile(obj, backend=backend, options=options)
            )

    if hasattr(pipe, "image_processor") and "image_processor" not in ignores:
        logger.info("Patching image_processor")

        from onediffx.utils.patch_image_processor import (
            patch_image_prcessor as patch_image_prcessor_,
        )

        patch_image_prcessor_(pipe.image_processor)

    return pipe


def fuse_qkv_projections_in_pipe(pipe):
    if hasattr(pipe, "fuse_qkv_projections"):
        pipe.fuse_qkv_projections()
    return pipe


def convert_pipe_to_memory_format(
    pipe, *, ignores=(), memory_format=torch.preserve_format
):
    import functools

    from nexfort.utils.attributes import multi_recursive_apply
    from nexfort.utils.memory_format import apply_memory_format

    if memory_format == torch.preserve_format:
        return pipe

    parts = [
        "unet",
        "controlnet",
        "fast_unet",  # for deepcache
        "prior",  # for StableCascadePriorPipeline
        "decoder",  # for StableCascadeDecoderPipeline
        "transformer",  # for Transformer-based DiffusionPipeline such as DiTPipeline and PixArtAlphaPipeline
        "vqgan",  # for StableCascadeDecoderPipeline
        "vae",
    ]
    multi_recursive_apply(
        pipe,
        parts,
        functools.partial(apply_memory_format, memory_format=memory_format),
        ignores=ignores,
        verbose=True,
    )
    return pipe


def save_pipe(pipe, dir="cached_pipe", *, ignores=(), overwrite=True):
    if not os.path.exists(dir):
        os.makedirs(dir)
    filtered_parts = _filter_parts(ignores=ignores)
    for part in filtered_parts:
        obj = _recursive_getattr(pipe, part, None)
        if (
            obj is not None
            and isinstance(obj, DeployableModule)
            and obj._deployable_module_dpl_graph is not None
            and obj.get_graph().is_compiled
        ):
            if not overwrite and os.path.isfile(os.path.join(dir, part)):
                logger.info(
                    f"Compiled graph already exists for {part}, not overwriting it."
                )
                continue
            logger.info(f"Saving {part}")
            obj.save_graph(os.path.join(dir, part))


def load_pipe(
    pipe,
    dir="cached_pipe",
    *,
    ignores=(),
):
    if not os.path.exists(dir):
        return
    filtered_parts = _filter_parts(ignores=ignores)
    for part in filtered_parts:
        obj = _recursive_getattr(pipe, part, None)
        if obj is not None and os.path.exists(os.path.join(dir, part)):
            logger.info(f"Loading {part}")
            obj.load_graph(os.path.join(dir, part))

    if "image_processor" not in ignores:
        logger.info("Patching image_processor")

        from onediffx.utils.patch_image_processor import (
            patch_image_prcessor as patch_image_prcessor_,
        )

        patch_image_prcessor_(pipe.image_processor)


def quantize_pipe(
    pipe, quant_submodules_config_path=None, top_percentage=90, *, ignores=(), **kwargs
):
    from nexfort.ao import quantize
    from nexfort.utils.attributes import multi_recursive_apply

    # if "dynamic_quant_filter_fn" not in kwargs:
    #     kwargs["dynamic_quant_filter_fn"] = dynamic_quant_filter_fn

    def load_quant_submodules_from_json(quant_submodules_config_path, top_percentage):
        with open(quant_submodules_config_path, "r") as file:
            data = json.load(file)
        submodules_with_ssim = [(fqn, details["ssim"]) for fqn, details in data.items()]
        submodules_with_ssim.sort(key=lambda x: x[1], reverse=True)
        top_n_percent_index = int(len(submodules_with_ssim) * (top_percentage / 100))
        top_submodules = [fqn for fqn, _ in submodules_with_ssim[:top_n_percent_index]]
        return top_submodules

    parts = [
        "unet",
        "controlnet",
        "fast_unet",  # for deepcache
        "prior",  # for StableCascadePriorPipeline
        "decoder",  # for StableCascadeDecoderPipeline
        "transformer",  # for Transformer-based DiffusionPipeline such as DiTPipeline and PixArtAlphaPipeline
    ]

    # def is_allowed_fqn(module, name, allowed_fqns=allowed_fqns):
    #     if allowed_fqns is None:
    #         return True
    #     return name in allowed_fqns

    if quant_submodules_config_path:
        allowed_fqns = load_quant_submodules_from_json(
            quant_submodules_config_path, top_percentage
        )
        quantization_function = functools.partial(
            quantize,
            quant_type=kwargs.pop("quant_type", None),
            filter_fn="is_allowed_fqn",
            filter_fn_kwargs={"allowed_fqns": allowed_fqns},
        )
        multi_recursive_apply(
            pipe, parts, quantization_function, ignores=ignores, verbose=True
        )
    else:
        multi_recursive_apply(
            pipe,
            parts,
            functools.partial(quantize, **kwargs),
            ignores=ignores,
            verbose=True,
        )

    return pipe
