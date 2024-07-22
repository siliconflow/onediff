"""[SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

## Performance Comparison

Updated on Mon 08 Apr 2024

Timings for 30 steps at 1024x1024
| Accelerator             | Baseline (non-optimized) | OneDiff(optimized) | OneDiff Quant(optimized) |
| ----------------------- | ------------------------ | ------------------ | ------------------------ |
| NVIDIA GeForce RTX 3090 | 8.03 s                   | 4.44 s ( ~44.7%)   | 3.34 s ( ~58.4%)         |

- torch   {version: 2.2.1+cu121}
- oneflow {git_commit: 710818c, version: 0.9.1.dev20240406+cu121, enterprise: True}

## Install

1. [OneDiff Installation Guide](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#install-onediff-enterprise)
2. [OneDiffx Installation Guide](https://github.com/siliconflow/onediff/tree/main/onediff_diffusers_extensions#install-and-setup)

## Usage:
> onediff/onediff_diffusers_extensions/examples/text_to_image_online_quant.py

```shell
# Baseline (non-optimized)
$   python text_to_image_online_quant.py \
        --model_id  /share_nfs/hf_models/stable-diffusion-xl-base-1.0  \
        --seed 1 \
        --backend torch  --height 1024 --width 1024 --output_file sdxl_torch.png
```
```shell
# OneDiff Quant(optimized)
$   python text_to_image_online_quant.py \
        --model_id  /share_nfs/hf_models/stable-diffusion-xl-base-1.0  \
        --seed 1 \
        --backend onediff \
        --cache_dir ./run_sdxl_quant \
        --height 1024 \
        --width 1024 \
        --output_file sdxl_quant.png   \
        --quantize \
        --conv_mae_threshold 0.1 \
        --linear_mae_threshold 0.2 \
        --conv_compute_density_threshold 900 \
        --linear_compute_density_threshold 300
```

| Option                                 | Range  | Default | Description                                                                  |
| -------------------------------------- | ------ | ------- | ---------------------------------------------------------------------------- |
| --conv_mae_threshold 0.1               | [0, 1] | 0.1     | MAE threshold for quantizing convolutional modules to 0.1.                   |
| --linear_mae_threshold 0.2             | [0, 1] | 0.2     | MAE threshold for quantizing linear modules to 0.2.                          |
| --conv_compute_density_threshold 900   | [0, ∞) | 900     | Computational density threshold for quantizing convolutional modules to 900. |
| --linear_compute_density_threshold 300 | [0, ∞) | 300     | Computational density threshold for quantizing linear modules to 300.        |

Notes:

1. Set CUDA device using export CUDA_VISIBLE_DEVICES=7.

2. The log *.pt file is cached. Quantization result information can be found in `cache_dir`/quantization_stats.json.

"""
import argparse
import time

import torch
from diffusers import AutoPipelineForText2Image
from onediff_quant.quantization import QuantizationConfig
from onediffx import compile_pipe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument(
        "--prompt", default="a photo of an astronaut riding a horse on mars"
    )
    parser.add_argument(
        "--output_file", default="astronaut_rides_horse_onediff_quant.png"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", default="onediff", choices=["onediff", "torch"])
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--cache_dir", default="./run_sd-v1-5")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--conv_mae_threshold", type=float, default=0.2)
    parser.add_argument("--linear_mae_threshold", type=float, default=0.4)
    parser.add_argument("--conv_compute_density_threshold", type=int, default=900)
    parser.add_argument("--linear_compute_density_threshold", type=int, default=300)
    return parser.parse_args()


def load_model(model_id):
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(f"cuda")
    return pipe


def compile_and_quantize_model(pipe, cache_dir, quantize, quant_params):
    pipe = compile_pipe(pipe)
    if quantize:
        config = QuantizationConfig.from_settings(
            **quant_params, cache_dir=cache_dir, plot_calibrate_info=True
        )
        pipe.unet.apply_online_quant(quant_config=config)
    return pipe


def save_image(image, output_file):
    image.save(output_file)
    print(f"Image saved to: {output_file}")


def main():
    args = parse_args()
    pipe = load_model(args.model_id)
    if args.backend == "onediff":
        compile_and_quantize_model(
            pipe,
            args.cache_dir,
            args.quantize,
            {
                "conv_mae_threshold": args.conv_mae_threshold,
                "linear_mae_threshold": args.linear_mae_threshold,
                "conv_compute_density_threshold": args.conv_compute_density_threshold,
                "linear_compute_density_threshold": args.linear_compute_density_threshold,
            },
        )
    torch.manual_seed(args.seed)
    # Warm-up
    pipe(prompt=args.prompt, num_inference_steps=1)

    # Run_inference
    for _ in range(5):
        start_time = time.time()
        torch.manual_seed(args.seed)
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
        ).images[0]
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")

    save_image(image, args.output_file)


if __name__ == "__main__":
    main()
