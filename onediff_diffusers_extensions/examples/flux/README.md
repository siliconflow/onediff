# Run Flux with onediff


## Environment setup

### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up compiler backend
Support two backends: oneflow and nexfort.

https://github.com/siliconflow/onediff?tab=readme-ov-file#install-a-compiler-backend

### Set up flux
HF model: https://huggingface.co/black-forest-labs/FLUX.1-dev  and https://huggingface.co/black-forest-labs/FLUX.1-schnell

HF pipeline: https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux

### Set up others
Install extra pkgs and set environment variable.
```bash
pip install --upgrade transformers
pip install --upgrade diffusers[torch]
pip install nvidia-cublas-cu12==12.4.5.8

export NEXFORT_FX_FORCE_TRITON_SDPA=1
```

## Run

### Run FLUX.1-dev 1024*1024 without compile (the original pytorch HF diffusers baseline)
```
python3 onediff_diffusers_extensions/examples/flux/text_to_image_flux.py \
--model black-forest-labs/FLUX.1-dev \
--height 1024 \
--width  1024 \
--steps 20 \
--seed 1 \
--output-image ./flux.png
```

### Run FLUX.1-dev 1024*1024 with compile [nexfort backend]

```
python3 onediff_diffusers_extensions/examples/flux/text_to_image_flux.py \
--model black-forest-labs/FLUX.1-dev \
--height 1024 \
--width  1024 \
--steps 20 \
--seed 1 \
--compiler nexfort \
--compiler-config '{"mode": "max-optimize:max-autotune:low-precision:cache-all", "memory_format": "channels_last"}' \
--output-image ./flux_nexfort_compile.png
```


### Run FLUX.1-schnell 1024*1024 without compile (the original pytorch HF diffusers baseline)
```
python3 onediff_diffusers_extensions/examples/flux/text_to_image_flux.py \
--model black-forest-labs/FLUX.1-schnell \
--height 1024 \
--width  1024 \
--steps 4 \
--seed 1 \
--output-image ./flux.png
```

### Run FLUX.1-schnell 1024*1024 with compile [nexfort backend]

```
python3 onediff_diffusers_extensions/examples/flux/text_to_image_flux.py \
--model black-forest-labs/FLUX.1-schnell \
--height 1024 \
--width  1024 \
--steps 4 \
--seed 1 \
--compiler nexfort \
--compiler-config '{"mode": "max-optimize:max-autotune:low-precision:cache-all", "memory_format": "channels_last"}' \
--output-image ./flux_nexfort_compile.png
```


## FLUX.1-dev Performance comparation
**Testing on NVIDIA H20-SXM4-80GB:**

Data update date: 2024-10-23

| Framework          | Iteration Speed (it/s) | E2E Time (seconds) | Max Memory Used (GiB) | Warmup time (seconds) <sup>1</sup> | Warmup with Cache time (seconds) |
|--------------------|------------------------|--------------------|-----------------------|-------------|------------------------|
| PyTorch            | 1.30                  | 15.72               | 35.73                 | 16.68       | -                      |
| OneDiff (NexFort)  | 1.76 (+35.4%)         | 11.57 (-26.4%)      | 34.85                | 750.78      | 28.57                  |

 <sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Platinum 8468V.

**Testing on NVIDIA L20-SXM4-48GB:**

Data update date: 2024-10-28

| Framework          | Iteration Speed (it/s) | E2E Time (seconds) | Max Memory Used (GiB) | Warmup time (seconds) <sup>2</sup> | Warmup with Cache time (seconds) |
|--------------------|------------------------|--------------------|-----------------------|-------------|------------------------|
| PyTorch            | 1.10                   | 18.45               | 35.71                | 18.695        | -                      |
| OneDiff (NexFort)  | 1.41 (+28.2%)         | 14.44 (-21.7%)      | 34.83                | 546.52      | 25.32                  |

 <sup>2</sup> OneDiff Warmup with Compilation time is tested on AMD EPYC 9354 32-Core Processor.



## FLUX.1-schnell Performance comparation
**Testing on NVIDIA H20-SXM4-80GB:**

Data update date: 2024-10-23

| Framework          | Iteration Speed (it/s) | E2E Time (seconds) | Max Memory Used (GiB) | Warmup time (seconds) <sup>1</sup> | Warmup with Cache time (seconds) |
|--------------------|------------------------|--------------------|-----------------------|-------------|------------------------|
| PyTorch            | 1.30             | 3.38              | 35.71               | 4.35      | -                      |
| OneDiff (NexFort)  | 1.75 (+34.6%)         | 2.46 (-27.2%)      | 34.83             | 201.41      | 19.57                 |

 <sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Platinum 8468V.

**Testing on NVIDIA L20-SXM4-48GB:**

Data update date: 2024-10-28

| Framework          | Iteration Speed (it/s) | E2E Time (seconds) | Max Memory Used (GiB) | Warmup time (seconds) <sup>2</sup> | Warmup with Cache time (seconds) |
|--------------------|------------------------|--------------------|-----------------------|-------------|------------------------|
| PyTorch            | 1.10                   | 3.94               | 35.69                | 4.15        | -                      |
| OneDiff (NexFort)  | 1.41 (+28.2%)         | 3.03 (-23.1%)      | 34.81                | 145.63      | 13.56                  |

 <sup>2</sup> OneDiff Warmup with Compilation time is tested on AMD EPYC 9354 32-Core Processor.
