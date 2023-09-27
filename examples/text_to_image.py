import torch 
from onediff.infer_compiler import oneflow_compile

from diffusers import StableDiffusionXLPipeline


model_id = "/share_nfs/hf_models/stable-diffusion-xl-base-1.0" 

pipe = StableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path = model_id,
    variant="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")
pipe.unet = oneflow_compile(pipe.unet)

prompt = "a photo of an astronaut riding a horse on mars"
# with flow.autocast("cuda"):
images = pipe(prompt).images
for i, image in enumerate(images):
    image.save(f"{prompt}-of-{i}.png")
