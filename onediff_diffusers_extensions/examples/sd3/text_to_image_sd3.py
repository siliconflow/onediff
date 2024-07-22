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
        default="stabilityai/stable-diffusion-3-medium-diffusers",
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
        default="photo of a dog and a cat both standing on a red box, with a blue ball in the middle with a parrot standing on top of the ball. The box has the text 'onediff'",
        help="Prompt for the image generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for the image generation.",
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
        default=4.5,
        help="The scale factor for the guidance.",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=28, help="Number of inference steps."
    )
    parser.add_argument(
        "--saved-image",
        type=str,
        default="./sd3.png",
        help="Path to save the generated image.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Seed for random number generation."
    )
    parser.add_argument(
        "--run_multiple_resolutions",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    parser.add_argument(
        "--run_multiple_prompts",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
    )
    return parser.parse_args()


args = parse_args()

device = torch.device("cuda")


def generate_texts(min_length=50, max_length=302):
    base_text = "a female character with long, flowing hair that appears to be made of ethereal, swirling patterns resembling the Northern Lights or Aurora Borealis. The background is dominated by deep blues and purples, creating a mysterious and dramatic atmosphere. The character's face is serene, with pale skin and striking features. She"

    additional_words = [
        "gracefully",
        "beautifully",
        "elegant",
        "radiant",
        "mysteriously",
        "vibrant",
        "softly",
        "gently",
        "luminescent",
        "sparkling",
        "delicately",
        "glowing",
        "brightly",
        "shimmering",
        "enchanting",
        "gloriously",
        "magnificent",
        "majestic",
        "fantastically",
        "dazzlingly",
    ]

    for i in range(min_length, max_length):
        idx = i % len(additional_words)
        base_text += " " + additional_words[idx]
        yield base_text


class SD3Generator:
    def __init__(self, model, compiler_config=None, quantize_config=None):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
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


def main():
    compiler_config = json.loads(args.compiler_config) if args.compiler_config else None
    quantize_config = json.loads(args.quantize_config) if args.quantize_config else None

    sd3 = SD3Generator(args.model, compiler_config, quantize_config)

    if args.run_multiple_prompts:
        # Note: diffusers will truncate the input prompt (limited to 77 tokens).
        # https://github.com/huggingface/diffusers/blob/8e1b7a084addc4711b8d9be2738441dfad680ce0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L238
        dynamic_prompts = generate_texts(max_length=101)
        prompt_list = list(dynamic_prompts)
    else:
        prompt_list = [args.prompt]

    gen_args = {
        "prompt": args.prompt,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
    }

    sd3.warmup(gen_args)

    for prompt in prompt_list:
        gen_args["prompt"] = prompt
        print(f"Processing prompt of length {len(prompt)} characters.")
        image, inference_time = sd3.generate(gen_args)
        assert inference_time < 20, "Prompt inference took too long"
        print(
            f"Generated image saved to {args.saved_image} in {inference_time:.2f} seconds."
        )
        cuda_mem_after_used = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Max used CUDA memory : {cuda_mem_after_used:.3f} GiB")

    if args.run_multiple_resolutions:
        gen_args["prompt"] = args.prompt
        print("Test run with multiple resolutions...")
        sizes = [1536, 1024, 768, 720, 576, 512, 256]
        for h in sizes:
            for w in sizes:
                gen_args["height"] = h
                gen_args["width"] = w
                print(f"Running at resolution: {h}x{w}")
                start_time = time.time()
                sd3.generate(gen_args)
                end_time = time.time()
                print(f"Inference time: {end_time - start_time:.2f} seconds")
                assert (
                    end_time - start_time
                ) < 20, "Resolution switch test took too long"


if __name__ == "__main__":
    main()
