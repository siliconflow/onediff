"""[Stable Diffusion V1.5 - Hugging Face Model Hub](https://huggingface.co/runwayml/stable-diffusion-v1-5)

## Performance Comparison

Updated on Tue 02 Apr 2024 

Timings for 50 steps at 512x512
| Accelerator             | Baseline (non-optimized) | OneDiff Quant(optimized) | Percentage improvement |
| ----------------------- | ------------------------ | ------------------------ | ---------------------- |
| NVIDIA GeForce RTX 3090 | 2.51 s                   | 0.92 s                   | ~63 %                  |

## Install

1. [OneDiff Installation Guide](https://github.com/siliconflow/onediff/tree/main?tab=readme-ov-file#installation)
2. [OneDiffx Installation Guide](https://github.com/siliconflow/onediff/tree/main/onediff_diffusers_extensions#install-and-setup)

## Usage:

```shell
# Baseline (non-optimized)
$   python text_to_image_online_quant.py \
        --model_id  /share_nfs/hf_models/stable-diffusion-v1-5  \
        --seed 1 \
        --backend torch  --height 512 --width 512 --output_file sd-v1-5_torch.png
```
```shell
# OneDiff Quant(optimized)
$   python text_to_image_online_quant.py \
        --model_id  /share_nfs/hf_models/stable-diffusion-v1-5  \
        --seed 1 \
        --backend onediff \
        --cache_dir ./run_sd-v1-5_quant \
        --height 512 \
        --width 512 \
        --output_file sd-v1-5_quant.png   \
        --quantize \
        --conv_mae_threshold 0.1 \
        --linear_mae_threshold 0.2 \
        --conv_compute_density_threshold 900 \
        --linear_compute_density_threshold 300
```

| Option                                 | Range  | Default | Description                                                                  |
| -------------------------------------- | ------ | ------- | ---------------------------------------------------------------------------- |
| --conv_mae_threshold 0.9               | [0, 1] | 0.1     | MAE threshold for quantizing convolutional modules to 0.1.                   |
| --linear_mae_threshold 1               | [0, 1] | 0.2     | MAE threshold for quantizing linear modules to 0.2.                            |
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
from onediffx import compile_pipe
from onediff_quant.quantization import QuantizationConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", default="a photo of an astronaut riding a horse on mars")
    parser.add_argument("--output_file", default="astronaut_rides_horse_onediff_quant.png")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", default="onediff", choices=["onediff", "torch"])
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--cache_dir", default="./run_sd-v1-5")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--conv_mae_threshold", type=float, default=0.2)
    parser.add_argument("--linear_mae_threshold", type=float, default=0.4)
    parser.add_argument("--conv_compute_density_threshold", type=int, default=900)
    parser.add_argument("--linear_compute_density_threshold", type=int, default=300)
    return parser.parse_args()

def load_model(model_id):
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.to(f"cuda")
    return pipe

def compile_and_quantize_model(pipe, cache_dir, quantize, quant_params):
    pipe = compile_pipe(pipe)
    if quantize:
        config = QuantizationConfig.from_settings(**quant_params, cache_dir=cache_dir, plot_calibrate_info=True)
        pipe.unet.apply_online_quant(quant_config=config)
    return pipe

def save_image(image, output_file):
    image.save(output_file)
    print(f"Image saved to: {output_file}")

def main():
    args = parse_args()
    pipe = load_model(args.model_id)
    if args.backend == "onediff": 
        compile_and_quantize_model(pipe, args.cache_dir, args.quantize, 
                                {"conv_mae_threshold": args.conv_mae_threshold,
                                    "linear_mae_threshold": args.linear_mae_threshold,
                                    "conv_compute_density_threshold": args.conv_compute_density_threshold,
                                    "linear_compute_density_threshold": args.linear_compute_density_threshold})
    
    # Warm-up
    pipe(prompt=args.prompt, num_inference_steps=1)

    # Run_inference
    for _ in range(4):
        start_time = time.time()
        torch.manual_seed(args.seed)
        image = pipe(prompt=args.prompt, height=args.height, width=args.width, num_inference_steps=args.num_inference_steps).images[0]
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")
    
    # [onediff_quant.png](https://github.com/siliconflow/onediff/assets/109639975/75cd9407-c9bb-423f-9e70-c15df76ff2b1)
    save_image(image, args.output_file)

if __name__ == "__main__":
    main()
