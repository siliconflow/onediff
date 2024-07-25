import argparse

import torch
from diffusers import AutoPipelineForText2Image

from onediff.quantization.quantize_pipeline import QuantPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--floatting_model_path", default="runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--prompt", default="a photo of an astronaut riding a horse on mars"
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--conv_compute_density_threshold", type=int, default=900)
    parser.add_argument("--linear_compute_density_threshold", type=int, default=300)
    parser.add_argument("--conv_ssim_threshold", type=float, default=0.985)
    parser.add_argument("--linear_ssim_threshold", type=float, default=0.991)
    parser.add_argument("--save_as_float", type=bool, default=False)
    parser.add_argument("--cache_dir", default="./run_sd-v1-5")
    parser.add_argument("--quantized_model", default="./quantized_model")
    return parser.parse_args()


args = parse_args()

pipe = QuantPipeline.from_pretrained(
    AutoPipelineForText2Image,
    args.floatting_model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")

pipe_kwargs = dict(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.num_inference_steps,
)

pipe.quantize(
    **pipe_kwargs,
    conv_compute_density_threshold=args.conv_compute_density_threshold,
    linear_compute_density_threshold=args.linear_compute_density_threshold,
    conv_ssim_threshold=args.conv_ssim_threshold,
    linear_ssim_threshold=args.linear_ssim_threshold,
    save_as_float=args.save_as_float,
    plot_calibrate_info=False,
    cache_dir=args.cache_dir
)

pipe.save_quantized(args.quantized_model, safe_serialization=True)
