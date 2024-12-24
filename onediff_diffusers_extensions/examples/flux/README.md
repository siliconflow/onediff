# Run Flux with nexfort backend (Beta Release)

1. [Environment Setup](#environment-setup)
   - [Set Up OneDiff](#set-up-onediff)
   - [Set Up NexFort Backend](#set-up-nexfort-backend)
   - [Set Up Diffusers Library](#set-up-diffusers)
   - [Download FLUX Model for Diffusers](#set-up-flux)
2. [Execution Instructions](#run)
3. [Performance Comparison](#performance-comparation)
4. [Dynamic Shape for Flux](#dynamic-shape-for-flux)
5. [Quality](#quality)

## Environment setup
### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

### Set up diffusers

```
# Ensure diffusers include the Flux pipeline.
pip3 install --upgrade diffusers[torch]
```
### Set up Flux
Model version for diffusers: https://huggingface.co/black-forest-labs/FLUX.1-dev

HF pipeline: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/flux.md

## Run

### Run 1024*1024 without compile (the original pytorch HF diffusers baseline)
```
python3 onediff_diffusers_extensions/examples/flux/text_to_image_flux.py \
    --saved-image flux.png
```

### Run 1024*1024 with compile


## Performance comparation
### Acceleration with Onediff-Community

```
NEXFORT_ENABLE_TRITON_AUTOTUNE_CACHE=0  \
NEXFORT_ENABLE_FP8_QUANTIZE_ATTENTION=0 \
python3 onediff_diffusers_extensions/examples/flux/text_to_image_flux.py \
    --transform \
    --saved-image flux_compile.png
```

Testing on NVIDIA H20, with image size of 1024*1024, iterating 20 steps:
| Metric                                           |                     |
| ------------------------------------------------ | ------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-11-13          |
| PyTorch iteration speed                          | 1.38 it/s           |
| OneDiff iteration speed                          | 1.89 it/s (+37.0%)  |
| PyTorch E2E time                                 | 14.94 s             |
| OneDiff E2E time                                 | 11.30 s (-24.4%)     |
| PyTorch Max Mem Used                             | 33.849 GiB          |
| OneDiff Max Mem Used                             | 33.850 GiB          |
| PyTorch Warmup with Run time                     | 16.15 s             |
| OneDiff Warmup with Compilation time<sup>1</sup> | 166.22 s            |
| OneDiff Warmup with Cache time                   | 12.58 s              |

<sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Platinum 8468V. Note this is just for reference, and it varies a lot on different CPU.

### Acceleration with Onediff-Enterprise(with quantization)
```
NEXFORT_FORCE_QUANTE_ON_CUDA=1 python3 onediff_diffusers_extensions/examples/flux/text_to_image_flux.py \
    --quantize \
    --transform \
    --saved-image flux_compile.png
```

Testing on NVIDIA H20, with image size of 1024*1024, iterating 20 steps:
| Metric                                           |                     |
| ------------------------------------------------ | ------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-11-13          |
| PyTorch iteration speed                          | 1.38 it/s           |
| OneDiff iteration speed                          | 2.98 it/s (+115.9%)  |
| PyTorch E2E time                                 | 14.94 s             |
| OneDiff E2E time                                 | 7.17 s (-52.0%)     |
| PyTorch Max Mem Used                             | 33.849 GiB          |
| OneDiff Max Mem Used                             | 22.879 GiB          |
| PyTorch Warmup with Run time                     | 16.15 s             |
| OneDiff Warmup with Compilation time<sup>1</sup> | 229.56 s            |
| OneDiff Warmup with Cache time                   | 8.28 s              |

<sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Platinum 8468V. Note this is just for reference, and it varies a lot on different CPU.

```
NEXFORT_FORCE_QUANTE_ON_CUDA=1 python3 onediff_diffusers_extensions/examples/flux/text_to_image_flux.py \
    --quantize \
    --transform \
    --speedup-t5 \  # Must quantize t5, because 4090 has only 24GB of memory
    --saved-image flux_compile.png
```


Testing on RTX 4090, with image size of 1024*1024, iterating 20 steps::
| Metric                                           |                     |
| ------------------------------------------------ | ------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-11-13          |
| PyTorch iteration speed                          | OOM                 |
| OneDiff iteration speed                          | 3.29 it/s           |
| PyTorch E2E time                                 | OOM                 |
| OneDiff E2E time                                 | 6.50 s              |
| PyTorch Max Mem Used                             | OOM                 |
| OneDiff Max Mem Used                             | 18.466 GiB          |
| PyTorch Warmup with Run time                     | OOM                 |
| OneDiff Warmup with Compilation time<sup>2</sup> | 169.16 s            |
| OneDiff Warmup with Cache time                   | 7.12 s             |

 <sup>2</sup> OneDiff Warmup with Compilation time is tested on AMD EPYC 7543 32-Core Processor


## Dynamic shape for Flux

Run:

```
python3 onediff_diffusers_extensions/examples/flux/text_to_image_flux.py \
    --quantize \
    --transform \
    --run_multiple_resolutions \
    --saved-image flux_compile.png
```

## Quality
When using nexfort as the backend for onediff compilation acceleration, the generated images are nearly lossless.(The following images are generated on an NVIDIA H20)

### Generated image with pytorch
<p align="center">
<img src="../../../imgs/flux_base.png">
</p>

### Generated image with nexfort acceleration(Community)
<p align="center">
<img src="../../../imgs/nexfort_flux_community.png">
</p>

### Generated image with nexfort acceleration(Enterprise)
<p align="center">
<img src="../../../imgs/nexfort_flux_enterprise.png">
</p>
