import hashlib
import os
from pathlib import Path

# ComfyUI
from folder_paths import get_input_directory

# onediff
from onediff import __version__ as onediff_version
from oneflow import __version__ as oneflow_version


def generate_short_sha256(string: str) -> str:
    return hashlib.sha256(string.encode("utf-8")).hexdigest()[:10]


def generate_graph_path(ckpt_name, model) -> Path:
    # get_save_graph_directory
    default_dir = get_input_directory()
    input_dir = os.getenv("COMFYUI_ONEDIFF_SAVE_GRAPH_DIR", default_dir)

    input_dir = Path(input_dir)
    graph_dir = input_dir / "graphs"

    file_name = f"{type(model).__name__}"

    return graph_dir / file_name
