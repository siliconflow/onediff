from onediff import OneFlowStableDiffusionInpaintPipeline

from diffusers.utils import load_image

import oneflow as flow
flow.mock_torch.enable()

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
init_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))
pipe = OneFlowStableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=flow.float16,
)
pipe = pipe.to("cuda")
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

with flow.autocast("cuda"):
    images = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
