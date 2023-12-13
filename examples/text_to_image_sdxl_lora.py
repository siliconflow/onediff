from packaging import version
import torch
import diffusers
from diffusers import DiffusionPipeline
from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.utils import TensorInplaceAssign


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=torch.float16
).to("cuda")
LORA_MODEL_ID = "hf-internal-testing/sdxl-1.0-lora"
LORA_FILENAME = "sd_xl_offset_example-lora_1.0.safetensors"

pipe.unet = oneflow_compile(pipe.unet)
pipe.load_lora_weights(LORA_MODEL_ID, weight_name=LORA_FILENAME)
generator = torch.manual_seed(0)

# The 'fuse_lora' API is not available in diffuser versions prior to 0.21.0.
if hasattr(pipe, "fuse_lora"):
    with TensorInplaceAssign(pipe.unet):
        pipe.fuse_lora(lora_scale=1.0)

if hasattr(pipe, "unfuse_lora"):
    with TensorInplaceAssign(pipe.unet):
        pipe.unfuse_lora()

# load LoRA twice to for checking result consistency
pipe.load_lora_weights(LORA_MODEL_ID, weight_name=LORA_FILENAME)
if hasattr(pipe, "fuse_lora"):
    with TensorInplaceAssign(pipe.unet):
        pipe.fuse_lora(lora_scale=1.0)

images_fusion = pipe(
    "masterpiece, best quality, mountain",
    generator=generator,
    height=1024,
    width=1024,
    num_inference_steps=30,
).images[0]

images_fusion.save("test_sdxl_lora.png")
