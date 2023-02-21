import oneflow as flow

flow.mock_torch.enable()
from onediff import OneFlowAltDiffusionPipeline


pipe = OneFlowAltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m9", torch_dtype=flow.float16)
pipe = pipe.to("cuda")
prompt = "黑暗精灵公主，非常详细，幻想，非常详细，数字绘画，概念艺术，敏锐的焦点，插图"
# prompt = "Astronaut riding a horse on Mars."
with flow.autocast("cuda"):
    images = pipe(prompt).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
