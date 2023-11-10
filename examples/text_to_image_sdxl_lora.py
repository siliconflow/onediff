from onediff.infer_compiler import oneflow_compile
from diffusers import DiffusionPipeline
import torch 

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16", torch_dtype=torch.float16).to("cuda")
lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

# TODO(): Not fused lora will affect the performance.
pipe.unet = oneflow_compile(pipe.unet)

generator = torch.manual_seed(0)
images_fusion = pipe(
    "masterpiece, best quality, mountain", generator=generator, num_inference_steps=2
).images
