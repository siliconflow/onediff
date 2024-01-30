from onediff.infer_compiler import oneflow_compile


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
    parts = [
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "unet",
        "controlnet",
        "fast_unet",  # for deepcache
        "vae.decoder",
        "vae.encoder",
    ]
    filtered_parts = []
    for part in parts:
        skip = False
        for ignore in ignores:
            if part == ignore or part.startswith(ignore + "."):
                skip = True
                break
        if not skip:
            filtered_parts.append(part)
    for part in filtered_parts:
        obj = recursive_getattr(pipe, part, None)
        if obj is not None:
            print(f"Compiling {part}")
            recursive_setattr(pipe, part, oneflow_compile(obj))
    return pipe


# TODO: Add save_pipe() and load_pipe()
