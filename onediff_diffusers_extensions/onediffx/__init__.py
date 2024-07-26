from onediff.infer_compiler import OneflowCompileOptions

from .compilers.diffusion_pipeline_compiler import (
    compile_pipe,
    load_pipe,
    quantize_pipe,
    save_pipe,
)

try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

__all__ = [
    "compile_pipe",
    "save_pipe",
    "load_pipe",
    "OneflowCompileOptions",
    "quantize_pipe",
]
