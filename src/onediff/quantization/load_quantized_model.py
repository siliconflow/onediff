import argparse

import torch
from diffusers import AutoPipelineForText2Image

from onediff.infer_compiler import oneflow_compile
from onediff.quantization.quantize_pipeline import QuantPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", default="a photo of an astronaut riding a horse on mars"
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--quantized_model", type=str, required=True)
    return parser.parse_args()


args = parse_args()

pipe = QuantPipeline.from_quantized(
    AutoPipelineForText2Image,
    args.quantized_model,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe = pipe.to("cuda")

pipe_kwargs = dict(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.num_inference_steps,
)

pipe.unet = oneflow_compile(pipe.unet)
pipe(**pipe_kwargs).images[0].save("test.png")
