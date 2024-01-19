import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline
from diffusers.utils import DIFFUSERS_CACHE
from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.utils import TensorInplaceAssign
from onediff.utils.lora import load_and_fuse_lora, unfuse_lora

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_ID = "/share_nfs/hf_models/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=torch.float16
).to("cuda")
# LORA_MODEL_ID = "hf-internal-testing/sdxl-1.0-lora"
# LORA_FILENAME = "sd_xl_offset_example-lora_1.0.safetensors"

LORA_MODEL_ID = "/data/home/wangyi/workspace/stable-diffusion-webui/models/Lora/watercolor_v1_sdxl_lora.safetensors"

pipe.unet = oneflow_compile(pipe.unet)
generator = torch.manual_seed(0)

# There are three methods to load LoRA into OneDiff compiled model
# 1. pipe.load_lora_weights (Low Performence)
# 2. pipe.load_lora_weights + TensorInplaceAssign + pipe.fuse_lora (Deprecated)
# 3. onediff.utils.load_and_fuse_lora (RECOMMENDED)


# 1. pipe.load_lora_weights (Low Performence)
# use load_lora_weights without fuse_lora is not recommended,
# due to the disruption of attention optimization, the inference speed is slowed down
pipe.load_lora_weights(LORA_MODEL_ID)
images_fusion = pipe(
    "masterpiece, best quality, mountain",
    generator=generator,
    height=1024,
    width=1024,
    num_inference_steps=30,
).images[0]
images_fusion.save("test_sdxl_lora_method1.png")
pipe.unload_lora_weights()


# need to rebuild UNet because method 1 has different computer graph with naive UNet
generator = torch.manual_seed(0)
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=torch.float16
).to("cuda")
pipe.unet = oneflow_compile(pipe.unet)
images_fusion = pipe(
    "masterpiece, best quality, mountain",
    generator=generator,
    height=1024,
    width=1024,
    num_inference_steps=30,
).images[0]
images_fusion.save("test_sdxl_lora.png")


# 2. pipe.load_lora_weights + TensorInplaceAssign + pipe.fuse_lora (Deprecated)
# The 'fuse_lora' API is not available in diffuser versions prior to 0.21.0.
generator = torch.manual_seed(0)
pipe.load_lora_weights(LORA_MODEL_ID)
if hasattr(pipe, "fuse_lora"):
    # TensorInplaceAssign is DEPRECATED and NOT RECOMMENDED, please use onediff.utils.load_and_fuse_lora
    with TensorInplaceAssign(pipe.unet):
        pipe.fuse_lora(lora_scale=1.0)
    images_fusion = pipe(
        "masterpiece, best quality, mountain",
        generator=generator,
        height=1024,
        width=1024,
        num_inference_steps=30,
    ).images[0]
    images_fusion.save("test_sdxl_lora_method2.png")

    with TensorInplaceAssign(pipe.unet):
        pipe.unfuse_lora()
pipe.unload_lora_weights()


# 3. onediff.utils.load_and_fuse_lora (RECOMMENDED)
# load_and_fuse_lora is equivalent to load_lora_weights + fuse_lora
generator = torch.manual_seed(0)
load_and_fuse_lora(pipe, LORA_MODEL_ID, lora_scale=1.0)
images_fusion = pipe(
    "masterpiece, best quality, mountain",
    generator=generator,
    height=1024,
    width=1024,
    num_inference_steps=30,
).images[0]

images_fusion.save("test_sdxl_lora_method3.png")

# 4. unfuse_lora can uninstall LoRA weights and restore the weights of UNet 
generator = torch.manual_seed(0)
unfuse_lora(pipe.unet)
images_fusion = pipe(
    "masterpiece, best quality, mountain",
    generator=generator,
    height=1024,
    width=1024,
    num_inference_steps=30,
).images[0]

images_fusion.save("test_sdxl_lora_without_lora.png")
