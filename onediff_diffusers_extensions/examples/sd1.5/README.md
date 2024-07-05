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
```
python3 benchmarks/text_to_image.py \
  --model /share_nfs/hf_models/stable-diffusion-v1-5 \
  --height 512 --width 512 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-v1-5.png \
  --prompt "product photography, world of warcraft orc warrior, white background" \
  --compiler none
```

### Run with compilation

```
python3 benchmarks/text_to_image.py \
    --model /share_nfs/hf_models/stable-diffusion-v1-5 \
    --height 512 --width 512 \
    --scheduler none \
    --steps 20 \
    --output-image ./stable-diffusion-v1-5-compile.png \
    --prompt "product photography, world of warcraft orc warrior, white background" \
    --compiler nexfort \
    --compiler-config '{"mode": "max-autotune:cudagraphs", "memory_format": "channels_last"}'
```

## Performance comparison

Testing on NVIDIA GeForce RTX 3090, with image size of 512*512, iterating 20 steps:
| Metric                                           |                                     |
| ------------------------------------------------ | ----------------------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-07-05                          |
| PyTorch iteration speed                          | 21.20 it/s                          |
| OneDiff iteration speed                          | 40.38 it/s (+90.5%)                 |
| PyTorch E2E time                                 | 1.07 s                              |
| OneDiff E2E time                                 | 0.56 s (-47.7%)                     |
| PyTorch Max Mem Used                             | 2.627 GiB                           |
| OneDiff Max Mem Used                             | 2.541 GiB                           |
| PyTorch Warmup with Run time                     |                               |
| OneDiff Warmup with Compilation time<sup>1</sup> |                            |
| OneDiff Warmup with Cache time                   |                              |

<!-- <sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Platinum 8468. Note this is just for reference, and it varies a lot on different CPU. -->

<!-- 
Testing on 4090:
| Metric                                           |                                     |
| ------------------------------------------------ | ----------------------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-06-29                          |
| PyTorch iteration speed                          | 6.67 it/s                           |
| OneDiff iteration speed                          | 11.51 it/s (+72.6%)                 |
| PyTorch E2E time                                 | 4.90 s                              |
| OneDiff E2E time                                 | 2.67 s (-45.5%)                     |
| PyTorch Max Mem Used                             | 18.799 GiB                          |
| OneDiff Max Mem Used                             | 17.902 GiB                          |
| PyTorch Warmup with Run time                     | 4.99 s                              |
| OneDiff Warmup with Compilation time<sup>2</sup> | 302.79 s                            |
| OneDiff Warmup with Cache time                   | 51.96 s                             |

 <sup>2</sup> AMD EPYC 7543 32-Core Processor -->


## Dynamic shape for SD1.5

 <!-- TODO -->

Run:

```
# The best practice mode configuration for dynamic shape is `max-optimize:max-autotune:low-precision`.
```

## Quality
When using nexfort as the backend for onediff compilation acceleration, the generated images are lossless.

<p align="center">
<img src="../../../imgs/nexfort_sd1-5_demo.png">
</p>
