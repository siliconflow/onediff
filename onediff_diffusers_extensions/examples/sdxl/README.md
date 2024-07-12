# Run SDXL with nexfort backend (Beta Release)

1. [Environment Setup](#environment-setup)
   - [Set Up OneDiff](#set-up-onediff)
   - [Set Up NexFort Backend](#set-up-nexfort-backend)
   - [Set Up Diffusers Library](#set-up-diffusers)
   - [Set Up SDXL](#set-up-sdxl)
2. [Execution Instructions](#run)
   - [Run Without Compilation (Baseline)](#run-without-compilation-baseline)
   - [Run With Compilation](#run-with-compilation)
3. [Performance Comparison](#performance-comparison)
4. [Dynamic Shape for SDXL](#dynamic-shape-for-sdxl)
5. [Quality](#quality)

## Environment setup
### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

### Set up diffusers

```
pip3 install --upgrade diffusers[torch]
```
### Set up SDXL
Model version for diffusers: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

HF pipeline: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_xl.md

## Run

### Run without compilation (Baseline)
```shell
python3 benchmarks/text_to_image.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-xl.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler none \
  --variant fp16 \
  --seed 1 \
  --print-output
```

### Run with compilation

```shell
python3 benchmarks/text_to_image.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-xl-compile.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler nexfort \
  --compiler-config '{"mode": "benchmark:cudagraphs:max-autotune:low-precision:cache-all", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": false, "overrides.conv_benchmark": true, "overrides.matmul_allow_tf32": true}}' \
  --variant fp16 \
  --seed 1 \
  --print-output
```

## Performance comparison

Testing on NVIDIA GeForce RTX 3090 / 4090, with image size of 1024*1024, iterating 20 steps:
| Metric                               | RTX 3090  1024*1024   | RTX 4090 1024*1024    |
| ------------------------------------ | --------------------- | --------------------- |
| Data update date (yyyy-mm-dd)        | 2024-07-10            | 2024-07-10            |
| PyTorch iteration speed              | 4.08 it/s             | 6.93 it/s             |
| OneDiff iteration speed              | 7.21 it/s (+76.7%)    | 13.92 it/s (+100.9%)  |
| PyTorch E2E time                     | 5.60 s                | 3.23 s                |
| OneDiff E2E time                     | 3.41 s (-39.1%)       | 1.67 s (-48.3%)       |
| PyTorch Max Mem Used                 | 10.467 GiB            | 10.467 GiB            |
| OneDiff Max Mem Used                 | 12.004 GiB            | 12.021 GiB            |
| PyTorch Warmup with Run time         |                       |                       |
| OneDiff Warmup with Compilation time | 474.36 s <sup>1</sup> | 236.54 s <sup>2</sup> |
| OneDiff Warmup with Cache time       | 306.84 s              | 104.57 s              |

<sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz. Note this is just for reference, and it varies a lot on different CPU.

<sup>2</sup> AMD EPYC 7543 32-Core Processor.


## Dynamic shape for SDXL

Run:

```shell
python3 benchmarks/text_to_image.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-xl-compile.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler nexfort \
  --compiler-config '{"mode": "cudagraphs:max-autotune:low-precision:cache-all", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": false, "overrides.conv_benchmark": true, "overrides.matmul_allow_tf32": true}, "dynamic": true}' \
  --run_multiple_resolutions 1
```

## Quality
When using nexfort as the backend for onediff compilation acceleration, the generated images are lossless.

<p align="center">
<img src="../../../imgs/nexfort_sdxl_demo.png">
</p>
