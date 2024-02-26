import hashlib
import os
from pathlib import Path
from oneflow import __version__ as oneflow_version

# ComfyUI
from folder_paths import get_input_directory

# onediff
from onediff import __version__ as onediff_version


def generate_short_sha256(string: str) -> str:
    return hashlib.sha256(string.encode("utf-8")).hexdigest()[:10]


def generate_graph_path(ckpt_name, model) -> Path:
    input_dir = os.getenv("COMFYUI_ONEDIFF_SAVE_GRAPH_DIR", get_input_directory())
    input_dir = Path(input_dir)
    graph_dir = input_dir / "graphs" / ckpt_name

    key = generate_short_sha256(f"{oneflow_version}{onediff_version}")

    file_name = f"{type(model).__name__}_{key}"

    return graph_dir / file_name
