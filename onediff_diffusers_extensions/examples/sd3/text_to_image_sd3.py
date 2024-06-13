import argparse
import json
import time

import torch
from diffusers import StableDiffusion3Pipeline
from onediffx import compile_pipe, quantize_pipe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use onediif (nexfort) to accelerate image generation with Stable Diffusion 3."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-3-medium",
        help="Model path or identifier.",
    )
    parser.add_argument(
        "--compiler-config", type=str, help="JSON string for compiler config."
    )
    parser.add_argument(
        "--quantize-config", type=str, help="JSON string for quantization config."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of a cat holding a sign that says hello world",
        help="Prompt for the image generation.",
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Height of the generated image."
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Width of the generated image."
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=28,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--saved-image",
        type=str,
        default="./sd3.png",
        help="Path to save the generated image.",
    )
    parser.add_argument(
        "--seed", type=int, default=333, help="Seed for random number generation."
    )
    return parser.parse_args()


args = parse_args()

device = torch.device("cuda")


class SD3Generator:
    def __init__(self, model, compiler_config=None, quantize_config=None):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model, torch_dtype=torch.float16, revision="refs/pr/26"
        )
        self.pipe.to(device)

        if compiler_config:
            print("compile...")
            self.pipe = self.compile_pipe(self.pipe, compiler_config)

        if quantize_config:
            print("quant...")
            self.pipe = self.quantize_pipe(self.pipe, quantize_config)

    def warmup(self, gen_args, warmup_iterations=1):
        warmup_args = gen_args.copy()

        warmup_args["generator"] = torch.Generator(device=device).manual_seed(0)

        print("Starting warmup...")
        for _ in range(warmup_iterations):
            self.pipe(**warmup_args)
        print("Warmup complete.")

    def generate(self, gen_args):
        self.warmup(gen_args)

        gen_args["generator"] = torch.Generator(device=device).manual_seed(args.seed)

        # Run the model
        start_time = time.time()
        images = self.pipe(**gen_args).images
        end_time = time.time()

        images[0].save(args.saved_image)

        return images[0], end_time - start_time

    def compile_pipe(self, pipe, compiler_config):
        options = compiler_config
        pipe = compile_pipe(
            pipe, backend="nexfort", options=options, fuse_qkv_projections=True
        )
        return pipe

    def quantize_pipe(self, pipe, quantize_config):
        pipe = quantize_pipe(pipe, ignores=[], **quantize_config)
        return pipe


def main():
    compiler_config = eval(args.compiler_config) if args.compiler_config else None
    quantize_config = eval(args.quantize_config) if args.quantize_config else None

    sd3 = SD3Generator(args.model, compiler_config, quantize_config)

    gen_args = {
        "prompt": args.prompt,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
    }

    image, inference_time = sd3.generate(gen_args)
    print(
        f"Generated image saved to {args.saved_image} in {inference_time:.2f} seconds."
    )


if __name__ == "__main__":
    main()
