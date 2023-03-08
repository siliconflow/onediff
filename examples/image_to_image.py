import oneflow as flow
from PIL import Image
flow.mock_torch.enable()
from onediff import OneFlowStableDiffusionImg2ImgPipeline

pipe = OneFlowStableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=flow.float16,
)

pipe = pipe.to("cuda")

prompt = "sea,beach,the waves crashed on the sand,blue sky whit white cloud"

img = Image.new("RGB", (512, 512), "#1f80f0")

with flow.autocast("cuda"):
    images = pipe(
        prompt,
        image=img,
        guidance_scale=10,
        num_inference_steps=100,
        compile_unet=False,
        output_type="np",
    ).images
    for i, image in enumerate(images):
        pipe.numpy_to_pil(image)[0].save(f"{prompt}-of-{i}.png")
