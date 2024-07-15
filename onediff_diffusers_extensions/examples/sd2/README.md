# Run SD2 with nexfort backend (Beta Release)

1. [Environment Setup](#environment-setup)
   - [Set Up OneDiff](#set-up-onediff)
   - [Set Up NexFort Backend](#set-up-nexfort-backend)
   - [Set Up Diffusers Library](#set-up-diffusers)
   - [Set Up SD2](#set-up-sd2)
2. [Execution Instructions](#run)
   - [Run Without Compilation (Baseline)](#run-without-compilation-baseline)
   - [Run With Compilation](#run-with-compilation)
3. [Performance Comparison](#performance-comparison)
4. [Dynamic Shape for SD2](#dynamic-shape-for-sd2)
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
### Set up SD2
Model version for diffusers: https://huggingface.co/stabilityai/stable-diffusion-2

HF pipeline: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_2.md

## Run

### Run without compilation (Baseline)
```shell
python3 benchmarks/text_to_image.py \
  --model stabilityai/stable-diffusion-2-1 \
  --height 768 --width 768 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-2-1.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler none \
  --print-output
```

### Run with compilation

```shell
python3 benchmarks/text_to_image.py \
  --model stabilityai/stable-diffusion-2-1 \
  --height 768 --width 768 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-2-1-compile.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler nexfort \
  --compiler-config '{"mode": "cudagraphs:benchmark:max-autotune:low-precision:cache-all", "memory_format": "channels_last", "options": {"triton.fuse_attention_allow_fp16_reduction": false, "inductor.optimize_linear_epilogue": false, "overrides.conv_benchmark": true, "overrides.matmul_allow_tf32": true}}' \
  --print-output
```

## Performance comparison

Testing on NVIDIA GeForce RTX 3090 / 4090, with image size of 786\*768 and 512\*512, iterating 20 steps:

| Metric                               | RTX3090, 768*768     | RTX3090, 512*512     | RTX4090, 768*768      | RTX4090, 512*512      |
| ------------------------------------ | -------------------- | -------------------- | --------------------- | --------------------- |
| Data update date (yyyy-mm-dd)        | 2024-07-10           | 2024-07-10           | 2024-07-10            | 2024-07-10            |
| PyTorch iteration speed              | 10.45 it/s           | 22.84 it/s           | 12.34 it/s            | 39.06 it/s            |
| OneDiff iteration speed              | 15.93 it/s (+52.4%)  | 44.84 it/s (+96.3%)  | 31.63 it/s (+156.3%)  | 83.63 it/s (+114.1%)  |
| PyTorch E2E time                     | 2.10 s               | 0.97 s               | 1.78s                 | 0.58 s                |
| OneDiff E2E time                     | 1.35 s (-35.7%)      | 0.49 s (-49.5%)      | 0.68s (-61.8%)        | 0.26 s (-55.2%)       |
| PyTorch Max Mem Used                 | 3.767 GiB            | 3.025 GiB            | 3.767 GiB             | 3.024 GiB             |
| OneDiff Max Mem Used                 | 3.558 GiB            | 3.018 GiB            | 3.567 GiB             | 3.016 GiB             |
| PyTorch Warmup with Run time         |                      |                      |                       |                       |
| OneDiff Warmup with Compilation time | 301.54 s<sup>1</sup> | 222.18 s<sup>1</sup> | 195.34 s <sup>2</sup> | 165.29 s <sup>1</sup> |
| OneDiff Warmup with Cache time       | 113.04 s             | 44.94 s              | 32.41 s               | 30.10 s               |

<sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz. Note this is just for reference, and it varies a lot on different CPU.

<sup>2</sup> AMD EPYC 7543 32-Core Processor.

## Dynamic shape for SD2

Run:

```shell
python3 benchmarks/text_to_image.py \
  --model stabilityai/stable-diffusion-2-1 \
  --height 768 --width 768 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-2-1-compile.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler nexfort \
  --compiler-config '{"mode": "cudagraphs:max-autotune:low-precision:cache-all", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": false, "overrides.conv_benchmark": true, "overrides.matmul_allow_tf32": true}, "dynamic": true}' \
  --run_multiple_resolutions 1
```

## Quality
When using nexfort as the backend for onediff compilation acceleration, the generated images are lossless.

<p align="center">
<img src="../../../imgs/nexfort_sd2_demo.png">
</p>
