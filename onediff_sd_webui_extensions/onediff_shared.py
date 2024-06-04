from typing import Dict

from compile.onediff_compiled_graph import OneDiffCompiledGraph

current_unet_graph = OneDiffCompiledGraph()
current_quantization = False
refiner_dict: Dict[str, str] = dict()
current_unet_type = {
    "is_sdxl": False,
    "is_sd2": False,
    "is_sd1": False,
    "is_ssd": False,
}
onediff_enabled = False
