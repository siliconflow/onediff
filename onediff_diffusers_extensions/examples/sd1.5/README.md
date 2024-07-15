# Run SD1.5 with nexfort backend (Beta Release)

1. [Environment Setup](#environment-setup)
   - [Set Up OneDiff](#set-up-onediff)
   - [Set Up NexFort Backend](#set-up-nexfort-backend)
   - [Set Up Diffusers Library](#set-up-diffusers)
   - [Set Up SD1.5](#set-up-sd15)
2. [Execution Instructions](#run)
   - [Run Without Compilation (Baseline)](#run-without-compilation-baseline)
   - [Run With Compilation](#run-with-compilation)
3. [Performance Comparison](#performance-comparison)
4. [Dynamic Shape for SD1.5](#dynamic-shape-for-sd15)
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
### Set up SD1.5
Model version for diffusers: https://huggingface.co/runwayml/stable-diffusion-v1-5

HF pipeline: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/stable_diffusion/overview.md

## Run

### Run without compilation (Baseline)
```shell
python3 benchmarks/text_to_image.py \
  --model runwayml/stable-diffusion-v1-5 \
  --height 512 --width 512 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-v1-5.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler none \
  --seed 1 \
  --print-output
```

### Run with compilation

```shell
python3 benchmarks/text_to_image.py \
  --model runwayml/stable-diffusion-v1-5 \
  --height 512 --width 512 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-v1-5-compile.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler nexfort \
  --compiler-config '{"mode": "cudagraphs:benchmark:max-autotune:low-precision:cache-all", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": false, "overrides.conv_benchmark": true, "overrides.matmul_allow_tf32": true}}' \
  --seed 1 \
  --print-output
```

## Performance comparison

Testing on NVIDIA GeForce RTX 3090 / 4090 / A100, with image size of 512*512, iterating 20 steps:
| Metric                               | RTX3090, 512*512      | RTX4090, 512*512      | A100-PCIE-40GB, 512*512 |
| ------------------------------------ | --------------------- | --------------------- | ----------------------- |
| Data update date (yyyy-mm-dd)        | 2024-07-10            | 2024-07-10            | 2024-07-15              |
| PyTorch iteration speed              | 21.20 it/s            | 34.46 it/s            | 36.24 it/s              |
| OneDiff iteration speed              | 48.00 it/s (+126.4%)  | 81.81 it/s (+137.4%)  | 67.29 it/s (+85.7 %)    |
| PyTorch E2E time                     | 1.07 s                | 0.67 s                | 0.65 s                  |
| OneDiff E2E time                     | 0.48 s (-55.1%)       | 0.28 s (-58.2%)       | 0.34 s (-47.7%)         |
| PyTorch Max Mem Used                 | 2.627 GiB             | 2.616 GiB             | 2.625 GiB               |
| OneDiff Max Mem Used                 | 2.587 GiB             | 2.709 GiB             | 2.727 GiB               |
| PyTorch Warmup with Run time         |                       |                       |                         |
| OneDiff Warmup with Compilation time | 233.61 s <sup>1</sup> | 177.32s <sup>2</sup>  | 197.51 s <sup>3</sup>   |
| OneDiff Warmup with Cache time       | 41.12 s               | 30.02s                | 27.01 s                 |

<sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz. Note this is just for reference, and it varies a lot on different CPU.

<sup>2</sup> AMD EPYC 7543 32-Core Processor.

<sup>3</sup> Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz.

## Dynamic shape for SD1.5

Run:

```shell
python3 benchmarks/text_to_image.py \
  --model runwayml/stable-diffusion-v1-5 \
  --height 512 --width 512 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-v1-5-compile.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler nexfort \
  --compiler-config '{"mode": "cudagraphs:max-autotune:low-precision:cache-all", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": false, "overrides.conv_benchmark": true, "overrides.matmul_allow_tf32": true}, "dynamic": true}' \
  --run_multiple_resolutions 1
```

## Quality
When using nexfort as the backend for onediff compilation acceleration, the generated images are lossless.

<p align="center">
<img src="../../../imgs/nexfort_sd1-5_demo.png">
</p>
