from typing import Dict

from compile.onediff_compiled_graph import OneDiffCompiledGraph

# from compile_utils import OneDiffCompiledGraph

current_unet_graph = OneDiffCompiledGraph()
graph_dict: Dict[str, OneDiffCompiledGraph] = dict()
refiner_dict: Dict[str, str] = dict()
current_unet_type = {
    "is_sdxl": False,
    "is_sd2": False,
    "is_sd1": False,
    "is_ssd": False,
}
onediff_enabled = True
