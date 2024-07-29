import os
from functools import wraps
from pathlib import Path
from textwrap import dedent
from zipfile import BadZipFile

import networks
import onediff_shared
from importlib_metadata import version

from onediff.utils.import_utils import is_oneflow_available

if is_oneflow_available():
    import oneflow as flow  # usort: skip

from compile import init_backend, is_oneflow_backend
from modules import shared
from modules.devices import torch_gc

from onediff.infer_compiler import DeployableModule

hints_message = dedent(
    """\
<div id="hintMessage" style="position: relative; padding: 20px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
    <button onclick="document.getElementById('hintMessage').style.display = 'none'" style="position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 18px; cursor: pointer;">&times;</button>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #31708f;">
        Hints Message
    </div>
    <div style="padding: 10px; border: 1px solid #31708f; border-radius: 5px; background-color: #f9f9f9;">
        Hints: Enterprise function is not supported on your system.
    </div>
    <p style="margin-top: 15px;">
        If you need Enterprise Level Support for your system or business, please send an email to
        <a href="mailto:business@siliconflow.com" style="color: #31708f; text-decoration: none;">business@siliconflow.com</a>.
        <br>
        Tell us about your use case, deployment scale, and requirements.
    </p>
    <p>
        <strong>GitHub Issue:</strong>
        <a href="https://github.com/siliconflow/onediff/issues" style="color: #31708f; text-decoration: none;">https://github.com/siliconflow/onediff/issues</a>
    </p>
</div>
"""
)

all_compiler_caches = []


def all_compiler_caches_path():
    import modules.shared as shared

    # for webui <= 1.8, the function will be called before ui initialized
    # At that time, shared opts doesn't have attribute `onediff_compiler_caches_path`
    if not hasattr(shared.opts, "onediff_compiler_caches_path"):
        return None

    caches_path = Path(shared.opts.onediff_compiler_caches_path)
    if not caches_path.exists():
        caches_path.mkdir(parents=True)
    return shared.opts.onediff_compiler_caches_path


def get_all_compiler_caches():
    global all_compiler_caches
    if len(all_compiler_caches) == 0:
        refresh_all_compiler_caches()
    return all_compiler_caches


def refresh_all_compiler_caches(path: Path = None):
    global all_compiler_caches
    path = path or all_compiler_caches_path()
    if path is None:
        return

    all_compiler_caches = [f.stem for f in Path(path).iterdir() if f.is_file()]


def check_structure_change(current_type: dict[str, bool], model):
    return current_type != get_model_type(model)


def load_graph(compiled_unet: DeployableModule, compiler_cache: str):
    from compile import OneDiffCompiledGraph

    if isinstance(compiled_unet, OneDiffCompiledGraph):
        compiled_unet = compiled_unet.graph_module

    if compiler_cache in [None, "None"]:
        return

    compiler_cache_path = all_compiler_caches_path() + f"/{compiler_cache}"
    if not Path(compiler_cache_path).exists():
        raise FileNotFoundError(
            f"Cannot find cache {compiler_cache_path}, please make sure it exists"
        )
    try:
        compiled_unet.load_graph(compiler_cache_path, run_warmup=True)
    except BadZipFile:
        raise RuntimeError(
            "Load cache failed. Please make sure that the --disable-safe-unpickle parameter is added when starting the webui"
        )
    except Exception as e:
        raise RuntimeError(
            f"Load cache failed ({e}). Please make sure cache has the same sd version (or unet architure) with current checkpoint"
        )
    return compiled_unet


def save_graph(compiled_unet: DeployableModule, saved_cache_name: str = ""):
    from compile import OneDiffCompiledGraph

    if isinstance(compiled_unet, OneDiffCompiledGraph):
        compiled_unet = compiled_unet.graph_module

    if saved_cache_name in ["", None]:
        return

    if not os.access(str(all_compiler_caches_path()), os.W_OK):
        raise PermissionError(
            f"The directory {all_compiler_caches_path()} does not have write permissions, and compiler cache cannot be written to this directory. \
                                Please change it in the settings to a directory with write permissions"
        )
    if not Path(all_compiler_caches_path()).exists():
        Path(all_compiler_caches_path()).mkdir()
    saved_cache_name = all_compiler_caches_path() + f"/{saved_cache_name}"
    if not Path(saved_cache_name).exists():
        compiled_unet.save_graph(saved_cache_name)


def onediff_enabled_decorator(func):
    @wraps(func)
    def wrapper(
        self,
        p,
        quantization=False,
        compiler_cache=None,
        saved_cache_name="",
        always_recompile=False,
        backend=None,
    ):
        onediff_shared.onediff_enabled = True
        if networks.originals is not None:
            networks.originals.undo()
        init_backend(backend)
        try:
            return func(
                self,
                p,
                quantization=quantization,
                compiler_cache=compiler_cache,
                saved_cache_name=saved_cache_name,
                always_recompile=always_recompile,
                backend=backend,
            )
        finally:
            if networks.originals is not None:
                networks.originals.__init__()
            onediff_shared.onediff_enabled = False
            onediff_shared.previous_unet_type.update(**get_model_type(shared.sd_model))
            onediff_gc()

    return wrapper


def singleton_decorator(func):
    has_been_called = False

    def wrapper(*args, **kwargs):
        nonlocal has_been_called
        if not has_been_called:
            has_been_called = True
            return func(*args, **kwargs)

    return wrapper


def get_model_type(model):
    return {
        "is_sdxl": model.is_sdxl,
        "is_sd2": model.is_sd2,
        "is_sd1": model.is_sd1,
        "is_ssd": model.is_ssd,
    }


def onediff_gc():
    torch_gc()
    if is_oneflow_backend():
        flow.cuda.empty_cache()


def varify_can_use_quantization():
    try:
        import oneflow

        if version("oneflow") < "0.9.1":
            return False
    except ImportError as e:
        return False
    try:
        import onediff_quant
    except ImportError as e:
        return False
    return hasattr(oneflow._C, "dynamic_quantization")
