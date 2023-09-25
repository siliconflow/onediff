from onediff.infer_compiler import oneflow_compile
from diffusers import StableDiffusionPipeline
import oneflow as flow

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=flow.float16,
)

pipe = pipe.to("cuda")
pipe.unet = oneflow_compile(pipe.unet)

prompt = "a photo of an astronaut riding a horse on mars"
with flow.autocast("cuda"):
    images = pipe(prompt).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
