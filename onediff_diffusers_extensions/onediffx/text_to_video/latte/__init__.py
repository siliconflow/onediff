# Latte originality is here https://github.com/Vchitect/Latte
# ```
# @article{ma2024latte,
#   title={Latte: Latent Diffusion Transformer for Video Generation},
#   author={Ma, Xin and Wang, Yaohui and Jia, Gengyun and Chen, Xinyuan and Liu, Ziwei and Li, Yuan-Fang and Chen, Cunjian and Qiao, Yu},
#   journal={arXiv preprint arXiv:2401.03048},
#   year={2024}
# }
# ```

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

if diffusers_version >= diffusers_0240_v:
    from .latte_t2v import LatteT2V
    from .pipeline_latte import LattePipeline
