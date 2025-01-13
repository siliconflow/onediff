# Run SD35 with nexfort backend (Beta Release)

1. [Environment Setup](#environment-setup)
   - [Set Up OneDiff](#set-up-onediff)
   - [Set Up NexFort Backend](#set-up-nexfort-backend)
   - [Set Up Diffusers Library](#set-up-diffusers)
   - [Download SD35 Model for Diffusers](#set-up-sd35)
2. [Execution Instructions](#run)
3. [Performance Comparison](#performance-comparation)
4. [Dynamic Shape for SD35](#dynamic-shape-for-sd25)
5. [Quality](#quality)

## Environment setup
### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

### Set up diffusers

```
# Ensure diffusers include the SD35 pipeline.
pip3 install --upgrade diffusers[torch]
```
### Set up SD35
Model version for diffusers: https://huggingface.co/stabilityai/stable-diffusion-3.5-large

HF pipeline: https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/README.md

## Run

### Run 1024*1024 without compile (the original pytorch HF diffusers baseline)
```
python3 onediff_diffusers_extensions/examples/sd35/text_to_image_sd35.py \
    --saved-image sd35.png
```

### Run 1024*1024 with compile


## Performance comparation
### Acceleration with Onediff-Community

```
NEXFORT_ENABLE_TRITON_AUTOTUNE_CACHE=0  \
NEXFORT_ENABLE_FP8_QUANTIZE_ATTENTION=0 \
python3 onediff_diffusers_extensions/examples/sd35/text_to_image_sd35.py \
    --transform \
    --saved-image sd35_compile.png
```

Testing on NVIDIA H20, with image size of 1024*1024, iterating 28 steps:
| Metric                                           |                     |
| ------------------------------------------------ | ------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-11-22          |
| PyTorch iteration speed                          | 1.47 it/s           |
| OneDiff iteration speed                          | 1.82 it/s (+23.8%)  |
| PyTorch E2E time                                 | 19.41 s             |
| OneDiff E2E time                                 | 15.99 s (-17.6%)    |
| PyTorch Max Mem Used                             | 28.525 GiB          |
| OneDiff Max Mem Used                             | 28.524 GiB          |
| PyTorch Warmup with Run time                     | 20.42 s             |
| OneDiff Warmup with Compilation time<sup>1</sup> | 96.81 s             |
| OneDiff Warmup with Cache time                   | 17.29 s             |

<sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Platinum 8468V. Note this is just for reference, and it varies a lot on different CPU.

### Acceleration with Onediff-Enterprise(with quantization)
```
NEXFORT_FORCE_QUANTE_ON_CUDA=1 python3 onediff_diffusers_extensions/examples/sd35/text_to_image_sd35.py \
    --quantize \
    --transform \
    --saved-image sd35_compile.png
```

Testing on NVIDIA H20, with image size of 1024*1024, iterating 28 steps:
| Metric                                           |                     |
| ------------------------------------------------ | ------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-11-22          |
| PyTorch iteration speed                          | 1.47 it/s           |
| OneDiff iteration speed                          | 2.72 it/s (+85.0%)  |
| PyTorch E2E time                                 | 19.41 s             |
| OneDiff E2E time                                 | 10.76 s (-44.6%)    |
| PyTorch Max Mem Used                             | 28.525 GiB          |
| OneDiff Max Mem Used                             | 20.713 GiB          |
| PyTorch Warmup with Run time                     | 20.42 s             |
| OneDiff Warmup with Compilation time<sup>1</sup> | 157.37 s            |
| OneDiff Warmup with Cache time                   | 12.04 s              |

<sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Platinum 8468V. Note this is just for reference, and it varies a lot on different CPU.

```
NEXFORT_FORCE_QUANTE_ON_CUDA=1 python3 onediff_diffusers_extensions/examples/sd35/text_to_image_sd35.py \
    --quantize \
    --transform \
    --speedup-t5 \  # Must quantize t5, because 4090 has only 24GB of memory
    --saved-image sd35_compile.png
```


Testing on RTX 4090, with image size of 1024*1024, iterating 28 steps::
| Metric                                           |                     |
| ------------------------------------------------ | ------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-11-22          |
| PyTorch iteration speed                          | OOM                 |
| OneDiff iteration speed                          | 3.01 it/s           |
| PyTorch E2E time                                 | OOM                 |
| OneDiff E2E time                                 | 9.79 s              |
| PyTorch Max Mem Used                             | OOM                 |
| OneDiff Max Mem Used                             | 20.109 GiB          |
| PyTorch Warmup with Run time                     | OOM                 |
| OneDiff Warmup with Compilation time<sup>2</sup> | 136.77 s            |
| OneDiff Warmup with Cache time                   | 10.74 s             |

 <sup>2</sup> OneDiff Warmup with Compilation time is tested on AMD EPYC 7543 32-Core Processor


## Dynamic shape for SD35

Run:

```
python3 onediff_diffusers_extensions/examples/sd35/text_to_image_sd35.py \
    --quantize \
    --transform \
    --run_multiple_resolutions \
    --saved-image sd35_compile.png
```

## Quality
When using nexfort as the backend for onediff compilation acceleration, the generated images are nearly lossless.(The following images are generated on an NVIDIA H20)

### Generated image with pytorch
<p align="center">
<img src="../../../imgs/sd35_base.png">
</p>

### Generated image with nexfort acceleration(Community)
<p align="center">
<img src="../../../imgs/nexfort_sd35_community.png">
</p>

### Generated image with nexfort acceleration(Enterprise)
<p align="center">
<img src="../../../imgs/nexfort_sd35_enterprise.png">
</p>
