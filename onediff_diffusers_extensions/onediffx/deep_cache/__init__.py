from packaging import version
import importlib
import importlib.metadata

diffusers_0193_v = version.parse("0.19.3")
diffusers_0240_v = version.parse("0.24.0")
diffusers_version = version.parse(importlib.metadata.version("diffusers"))

if diffusers_version < diffusers_0193_v:
    raise ImportError(
        f"onediffx supports at least version of diffusers-0.19.3, Currently version {str(diffusers_version)}! Please upgrade diffusers!"
    )

from .models.pipeline_utils import disable_deep_cache_pipeline

from .pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from .pipeline_stable_diffusion import StableDiffusionPipeline

if diffusers_version >= diffusers_0240_v:
    from .pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
