import os
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from zipfile import BadZipFile

import onediff_shared

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
    all_compiler_caches = [f.stem for f in Path(path).iterdir() if f.is_file()]


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


@contextmanager
def onediff_enabled():
    onediff_shared.onediff_enabled = True
    try:
        yield
    finally:
        onediff_shared.onediff_enabled = False
