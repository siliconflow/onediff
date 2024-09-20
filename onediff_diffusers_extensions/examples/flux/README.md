# Run FLUX with nexfort backend (Beta Release)

1. [Environment Setup](#environment-setup)
   - [Set Up OneDiff](#set-up-onediff)
   - [Set Up NexFort Backend](#set-up-nexfort-backend)
   - [Set Up Diffusers Library](#set-up-diffusers)
   - [Set Up FLUX](#set-up-flux)
2. [Execution Instructions](#run)
   - [Run Without Compilation (Baseline)](#run-without-compilation-baseline)
   - [Run With Compilation](#run-with-compilation)
3. [Performance Comparison](#performance-comparison)
4. [Dynamic Shape for FLUX](#dynamic-shape-for-flux)

## Environment setup
### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

### Set up diffusers

```
pip3 install --upgrade diffusers[torch]
```
### Set up FLUX
Model version for diffusers: https://huggingface.co/black-forest-labs/FLUX.1-schnell

HF pipeline: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/flux.md

## Run

### Run without compilation (Baseline)
```shell
python3 benchmarks/text_to_image.py \
  --model black-forest-labs/FLUX.1-schnell \
  --height 1024 --width 1024 \
  --scheduler none \
  --steps 4 \
  --output-image ./flux-schnell.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler none \
  --dtype bfloat16 \
  --seed 1 \
  --print-output
```

### Run with compilation

```shell
python3 benchmarks/text_to_image.py \
  --model black-forest-labs/FLUX.1-schnell \
  --height 1024 --width 1024 \
  --scheduler none \
  --steps 4 \
  --output-image ./flux-schnell-compile.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler nexfort \
  --compiler-config '{"mode": "benchmark:cudagraphs:max-autotune:low-precision:cache-all", "memory_format": "channels_last", "options": {"cuda.fuse_timestep_embedding": false, "inductor.force_triton_sdpa": true}}' \
  --dtype bfloat16 \
  --seed 1 \
  --print-output
```

## Performance comparison

Testing on NVIDIA A800-SXM4-80GB, with image size of 1024*1024, iterating 4 steps:
| Metric                               | A800-SXM4-80GB 1024*1024 |
| ------------------------------------ | ------------------------ |
| Data update date (yyyy-mm-dd)        | 2024-08-07               |
| PyTorch iteration speed              | 2.18 it/s                |
| OneDiff iteration speed              | 2.80 it/s (+28.4%)       |
| PyTorch E2E time                     | 2.06 s                   |
| OneDiff E2E time                     | 1.53 s (-25.7%)          |
| PyTorch Max Mem Used                 | 35.79 GiB                |
| OneDiff Max Mem Used                 | 40.44 GiB                |
| PyTorch Warmup with Run time         | 2.81 s                   |
| OneDiff Warmup with Compilation time | 253.01 s                 |
| OneDiff Warmup with Cache time       | 73.63 s                  |

<sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz. Note this is just for reference, and it varies a lot on different CPU.


## Dynamic shape for FLUX

Run:

```shell
python3 benchmarks/text_to_image.py \
  --model black-forest-labs/FLUX.1-schnell \
  --height 1024 --width 1024 \
  --scheduler none \
  --steps 4 \
  --output-image ./flux-schnell-compile.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler nexfort \
  --compiler-config '{"mode": "benchmark:cudagraphs:max-autotune:low-precision:cache-all", "memory_format": "channels_last", "options": {"cuda.fuse_timestep_embedding": false, "inductor.force_triton_sdpa": true}, "dynamic", true}' \
  --run_multiple_resolutions 1 \
  --dtype bfloat16 \
  --seed 1 \
```
