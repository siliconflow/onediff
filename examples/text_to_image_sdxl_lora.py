from onediff.infer_compiler import oneflow_compile
from diffusers import DiffusionPipeline
import torch

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=torch.float16
).to("cuda")
LORA_MODEL_ID = "hf-internal-testing/sdxl-1.0-lora"
LORA_FILENAME = "sd_xl_offset_example-lora_1.0.safetensors"
pipe.load_lora_weights(LORA_MODEL_ID, weight_name=LORA_FILENAME)

# TODO(): Not fused lora will affect the performance.
pipe.unet = oneflow_compile(pipe.unet)

generator = torch.manual_seed(0)
images_fusion = pipe(
    "masterpiece, best quality, mountain",
    generator=generator,
    height=1024,
    width=1024,
    num_inference_steps=30,
).images[0]
images_fusion.save("test_sdxl_lora.png")
