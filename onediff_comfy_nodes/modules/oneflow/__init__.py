from .booster_basic import BasicOneFlowBoosterExecutor
from .booster_deepcache import DeepcacheBoosterExecutor
from .booster_patch import PatchBoosterExecutor
from .config import _USE_UNET_INT8, ONEDIFF_QUANTIZED_OPTIMIZED_MODELS
from .patch_management.patch_for_oneflow import *

from .hijack_animatediff import animatediff_hijacker
from .hijack_comfyui_instantid import comfyui_instantid_hijacker
from .hijack_ipadapter_plus import ipadapter_plus_hijacker
from .hijack_model_management import model_management_hijacker
from .hijack_model_patcher import model_patch_hijacker
from .hijack_nodes import nodes_hijacker
from .hijack_pulid_comfyui import pulid_comfyui_hijacker
from .hijack_samplers import samplers_hijack
from .hijack_utils import comfy_utils_hijack

model_management_hijacker.hijack()  # add flow.cuda.empty_cache()
nodes_hijacker.hijack()
samplers_hijack.hijack()
animatediff_hijacker.hijack()
ipadapter_plus_hijacker.hijack()
comfyui_instantid_hijacker.hijack()
model_patch_hijacker.hijack()
comfy_utils_hijack.hijack()
pulid_comfyui_hijacker.hijack()
