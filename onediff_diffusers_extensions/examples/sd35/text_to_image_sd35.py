import argparse
import json
import time

import nexfort

import torch
from diffusers import StableDiffusion3Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use nexfort to accelerate image generation with SD35."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-3.5-large",
        help="Model path or identifier.",
    )
    parser.add_argument(
        "--speedup-t5",
        action="store_true",
        help="Enable optimize t5.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable fp8 quantization.",
    )
    parser.add_argument(
        "--transform",
        action="store_true",
        help="Enable speedup with nexfort.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="evening sunset scenery blue sky nature, glass bottle with a galaxy in it.",
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
        default=3.5,
        help="The scale factor for the guidance.",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=28, help="Number of inference steps."
    )
    parser.add_argument(
        "--saved-image",
        type=str,
        default="./sd35.png",
        help="Path to save the generated image.",
    )
    parser.add_argument(
        "--seed", type=int, default=20, help="Seed for random number generation."
    )
    parser.add_argument(
        "--run_multiple_resolutions",
        action="store_true",
    )
    parser.add_argument(
        "--run_multiple_prompts",
        action="store_true",
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


class SD35Generator:
    def __init__(
        self,
        model,
        enable_quantize=False,
        enable_fast_transformer=False,
        enable_speedup_t5=False,
    ):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
        )

        # Put the quantize process after `self.pipe.to(device)` if you have more than 32GB ram.
        if enable_quantize:
            print("quant...")
            from nexfort.quantization import quantize

            self.pipe.transformer = quantize(
                self.pipe.transformer, quant_type="fp8_e4m3_e4m3_dynamic_per_tensor"
            )
            if enable_speedup_t5:
                self.pipe.text_encoder_2 = quantize(
                    self.pipe.text_encoder_2,
                    quant_type="fp8_e4m3_e4m3_dynamic_per_tensor",
                )

        self.pipe.to(device)

        if enable_fast_transformer:
            print("compile...")
            from nexfort.compilers import transform

            self.pipe.transformer = transform(self.pipe.transformer)
            if enable_speedup_t5:
                self.pipe.text_encoder_2 = transform(self.pipe.text_encoder_2)

    def warmup(self, gen_args, warmup_iterations=1):
        warmup_args = gen_args.copy()

        # warmup_args["generator"] = torch.Generator(device=device).manual_seed(0)
        torch.manual_seed(args.seed)

        print("Starting warmup...")
        start_time = time.time()
        for _ in range(warmup_iterations):
            self.pipe(**warmup_args)
        end_time = time.time()
        print("Warmup complete.")
        print(f"Warmup time: {end_time - start_time:.2f} seconds")

    def generate(self, gen_args):
        # gen_args["generator"] = torch.Generator(device=device).manual_seed(args.seed)
        torch.manual_seed(args.seed)

        # Run the model
        start_time = time.time()
        image = self.pipe(**gen_args).images[0]
        end_time = time.time()

        image.save(args.saved_image)

        return image, end_time - start_time


def main():
    sd35 = SD35Generator(args.model, args.quantize, args.transform, args.speedup_t5)

    if args.run_multiple_prompts:
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
    }

    sd35.warmup(gen_args)

    for prompt in prompt_list:
        gen_args["prompt"] = prompt
        print(f"Processing prompt of length {len(prompt)} characters.")
        image, inference_time = sd35.generate(gen_args)
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
                sd35.generate(gen_args)
                end_time = time.time()
                print(f"Inference time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
