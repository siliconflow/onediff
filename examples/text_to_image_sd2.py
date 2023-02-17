import oneflow as flow

# monkey patch hacks
flow.strided = None
flow_randn = flow.randn


def dummy_randn(*args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if not k == "layout"}
    return flow_randn(*args, **kwargs)


flow.randn = dummy_randn

flow.mock_torch.enable()
from diffusers import EulerDiscreteScheduler
from onediff import OneFlowStableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2"
# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = OneFlowStableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, revision="fp16", torch_dtype=flow.float16
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, height=768, width=768).images[0]

image.save("astronaut_rides_horse.png")
