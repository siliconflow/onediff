from onediff import OneFlowStableDiffusionPipeline

import oneflow as flow
flow.mock_torch.enable()

pipe = OneFlowStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=flow.float16,
)

pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with flow.autocast("cuda"):
    images = pipe(prompt).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
