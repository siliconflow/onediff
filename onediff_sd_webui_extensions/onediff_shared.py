from compile import OneDiffCompiledGraph

current_unet_graph = OneDiffCompiledGraph()
current_quantization = False
previous_unet_type = {
    "is_sdxl": False,
    "is_sd2": False,
    "is_sd1": False,
    "is_ssd": False,
}
onediff_enabled = False
onediff_backend = None

# controlnet
controlnet_enabled = False
controlnet_compiled = False
previous_is_controlnet = False
