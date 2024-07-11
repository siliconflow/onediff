import argparse
import json
import time

from diffusers import DPMSolverMultistepScheduler, KolorsPipeline
from onediffx import compile_pipe, quantize_pipe
from onediff.infer_compiler import oneflow_compile
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use onediif to accelerate image generation with Kolors"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Kwai-Kolors/Kolors-diffusers",
        help="Model path or identifier.",
    )
    parser.add_argument(
        "--compiler",
        type=str,
        default="none",
        help="Compiler backend to use. Options: 'none', 'nexfort', 'oneflow'",
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
        default='一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着"可图"',
        help="Prompt for the image generation.",
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Height of the generated image."
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Width of the generated image."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.5,
        help="The scale factor for the guidance.",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=50, help="Number of inference steps."
    )
    parser.add_argument(
        "--saved-image",
        type=str,
        default="./kolors.png",
        help="Path to save the generated image.",
    )
    parser.add_argument(
        "--seed", type=int, default=66, help="Seed for random number generation."
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Number of warm-up iterations before actual inference.",
    )
    return parser.parse_args()


args = parse_args()

device = torch.device("cuda")


class KolorsGenerator:
    def __init__(
        self, model, compiler_config=None, quantize_config=None, compiler="none"
    ):
        self.pipe = KolorsPipeline.from_pretrained(
            model, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, use_karras_sigmas=True
        )

        if compiler == "nexfort":
            if compiler_config:
                print("nexfort backend compile...")
                self.pipe = self.compile_pipe(self.pipe, compiler_config)

            if quantize_config:
                print("nexfort backend quant...")
                self.pipe = self.quantize_pipe(self.pipe, quantize_config)
        elif compiler == "oneflow":
            print("oneflow backend compile...")
            self.pipe.unet = self.oneflow_compile(self.pipe.unet)

    def warmup(self, gen_args, warmup_iterations):
        warmup_args = gen_args.copy()

        warmup_args["generator"] = torch.Generator(device=device).manual_seed(0)

        print("Starting warmup...")
        start_time = time.time()
        for _ in range(warmup_iterations):
            self.pipe(**warmup_args)
        end_time = time.time()
        print("Warmup complete.")
        print(f"Warmup time: {end_time - start_time:.2f} seconds")

    def generate(self, gen_args):
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

    def oneflow_compile(self, unet):
        return oneflow_compile(unet)


def main():
    nexfort_compiler_config = (
        json.loads(args.compiler_config) if args.compiler_config else None
    )
    nexfort_quantize_config = (
        json.loads(args.quantize_config) if args.quantize_config else None
    )

    kolors = KolorsGenerator(
        args.model,
        nexfort_compiler_config,
        nexfort_quantize_config,
        compiler=args.compiler,
    )

    gen_args = {
        "prompt": args.prompt,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
    }

    kolors.warmup(gen_args, args.warmup_iterations)

    image, inference_time = kolors.generate(gen_args)
    print(
        f"Generated image saved to {args.saved_image} in {inference_time:.2f} seconds."
    )
    cuda_mem_after_used = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Max used CUDA memory : {cuda_mem_after_used:.3f}GiB")


if __name__ == "__main__":
    main()
