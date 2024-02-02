from pathlib import Path
from onediff.infer_compiler import oneflow_compile

COMPILED_PARTS = [
    "text_encoder",
    "text_encoder_2",
    "image_encoder",
    "unet",
    "controlnet",
    "fast_unet",  # for deepcache
    "vae.decoder",
    "vae.encoder",
]


def filter_parts(ignores=()):
    filtered_parts = []
    for part in COMPILED_PARTS:
        skip = False
        for ignore in ignores:
            if part == ignore or part.startswith(ignore + "."):
                skip = True
                break
        if not skip:
            filtered_parts.append(part)

    return filtered_parts


def recursive_getattr(obj, attr, default=None):
    attrs = attr.split(".")
    for attr in attrs:
        if not hasattr(obj, attr):
            return default
        obj = getattr(obj, attr, default)
    return obj


def recursive_setattr(obj, attr, value):
    attrs = attr.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


def compile_pipe(
    pipe,
    *,
    ignores=(),
):
    for part in filter_parts(ignores):
        obj = recursive_getattr(pipe, part, None)
        if obj is not None:
            print(f"Compiling {part}")
            recursive_setattr(pipe, part, oneflow_compile(obj))

    if "image_processor" not in ignores:
        print("Patching image_processor")

        from onediffx.utils.patch_image_processor import (
            patch_image_prcessor as patch_image_prcessor_,
        )

        patch_image_prcessor_(pipe.image_processor)

    return pipe


def save_pipe(
    pipe,
    graphs_dir,
    *,
    ignores=(),
    overwrite: bool = True,
):
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    for part in filter_parts(ignores):
        obj = recursive_getattr(pipe, part, None)
        if obj is None:
            continue

        graph_path = (graphs_dir / part.replace(".", "_")).with_suffix(".of")
        if not overwrite and graph_path.exists():
            print(f"Compiled graph already exists for {part}, not overwriting it.")
            continue

        print(f"Saving compiled graph for {part}")
        obj.save_graph(graph_path)


def load_pipe(
    pipe,
    graphs_dir,
    *,
    ignores=(),
):
    graphs_dir = Path(graphs_dir)

    for part in filter_parts(ignores):
        obj = recursive_getattr(pipe, part, None)
        if obj is None:
            continue

        graph_path = (graphs_dir / part.replace(".", "_")).with_suffix(".of")
        if not graph_path.exists():
            print(f"No compiled graph found for {part}, skipping it.")
            continue

        print(f"Loading compiled graph for {part}")
        obj.load_graph(graph_path)

    return pipe
