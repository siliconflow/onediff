import sys
from pathlib import Path
from contextlib import contextmanager
sys.path.append("../")
from config import COMFYUI_ROOT

import argparse

import torch
from nodes import (CheckpointLoaderSimple, CLIPTextEncode, EmptyLatentImage,
                   KSampler, SaveImage, VAEDecode)
from onediff_comfy_nodes import OneDiffCheckpointLoaderSimple
import time



def checkpoint_loader_factory(checkpoint_path, use_onediff=False):
    if use_onediff:
        return OneDiffCheckpointLoaderSimple().onediff_load_checkpoint(checkpoint_path, vae_speedup="enable")
    else:
        return CheckpointLoaderSimple().load_checkpoint(checkpoint_path)

@contextmanager
def cost_time():
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"cost time: {end_time - start_time}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive",
        type=str,
        default="beautiful scenery nature glass bottle landscape, , purple galaxy bottle,",
    )
    parser.add_argument("--negative", type=str, default="text, watermark")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--use_onediff", action="store_true")
    args = parser.parse_args()
    return args


def display_gpu_memory():
    mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
    print(f"GPU memory used: {mem}")

@torch.no_grad()
def pipeline(
    use_onediff=False,
    checkpoint_path="sd_xl_base_1.0.safetensors",
    positive="beautiful scenery nature glass bottle landscape, , purple galaxy bottle,",
    negative="text, watermark",
    steps=20,
    seed=1,
    height=1024,
    width=1024,
    warmup_steps=0,
    *args,
    **kwargs
):

    model_patcher, clip, vae = checkpoint_loader_factory(checkpoint_path, use_onediff)

    positive = CLIPTextEncode().encode(clip, positive)[0]
    negative = CLIPTextEncode().encode(clip, negative)[0]
      
    latent = EmptyLatentImage().generate(height=height, width=width, batch_size=1)[0]

    torch.manual_seed(seed)
    ksampler, vae_decode, save_image = KSampler(), VAEDecode(), SaveImage()

    input_dict = {
        "model": model_patcher,
        "seed": seed,
        "steps": steps,
        "cfg": 8.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "positive": positive,
        "negative": negative,
        "latent_image": latent,
        "denoise": 1.0,
    }

    # Image generation and saving
    for _ in range(warmup_steps + 1):  # Including warm-up steps and saving step
        with cost_time():
            samples = ksampler.sample(**input_dict)[0]
            generated_image = vae_decode.decode(vae, samples)[0]
            saved_image_info = save_image.save_images(generated_image)

    # Print image path
    image_path = Path(COMFYUI_ROOT) / "output" / saved_image_info["ui"]["images"][0]["filename"]
    print(f"Saved image to {image_path}")

    torch.cuda.empty_cache()
    return image_path


if __name__ == "__main__":
    args = parse_args()
    pipeline(**vars(args))