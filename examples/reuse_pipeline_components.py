"""
This example shows how to reuse the components of a pipeline to create new pipelines.

Usage:
    $ python reuse_pipeline_components.py --ckpt_path <ckpt_path> 
"""
import PIL
import argparse
import requests
import torch
from io import BytesIO
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from onediff.infer_compiler import oneflow_compile


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    return parser.parse_args()


args = get_args()


def initialize_pipelines(ckpt_path, model_params):
    pipeline = StableDiffusionPipeline.from_pretrained(
        ckpt_path, device="cpu", **model_params
    )

    device = torch.device("cuda:7")

    pipeline.to(device)
    pipeline.unet = oneflow_compile(pipeline.unet)
    img2img_pipe = StableDiffusionImg2ImgPipeline(**pipeline.components)
    inpaint_pipe = StableDiffusionInpaintPipeline(**pipeline.components)
    return pipeline, img2img_pipe, inpaint_pipe


def download_image(url, resize=None):
    response = requests.get(url)
    image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
    if resize:
        image = image.resize(resize)
    return image


def inference_text2img(pipe):
    prompt = "A fantasy landscape, trending on artstation"
    result_image = pipe(prompt=prompt).images[0]
    result_image.save("inference_text2img.png")


def inference_img2img(pipe):
    img_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    init_image = download_image(img_url, resize=(768, 512))
    prompt = "A fantasy landscape, trending on artstation"
    result_image = pipe(
        prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5
    ).images[0]
    result_image.save("inference_img2img.png")


def inference_inpaint(pipe):
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    init_image = download_image(img_url, resize=(512, 512))
    mask_image = download_image(mask_url, resize=(512, 512))

    prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
    result_image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[
        0
    ]
    result_image.save("inference_inpaint.png")


def main(ckpt_path, model_params):
    text2img, img2img, inpaint = initialize_pipelines(ckpt_path, model_params)

    inference_text2img(text2img)
    inference_img2img(img2img)
    inference_inpaint(inpaint)


if __name__ == "__main__":
    ckpt_path = args.ckpt_path
    model_params = {"torch_dtype": torch.float16}
    main(ckpt_path, model_params)
