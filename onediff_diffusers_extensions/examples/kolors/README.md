# Run Kolors with onediff


## Environment setup

### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up compiler backend
Support two backends: oneflow and nexfort.

https://github.com/siliconflow/onediff?tab=readme-ov-file#install-a-compiler-backend


### Set up diffusers

```
# Ensure diffusers include the kolors pipeline.
pip install git+https://github.com/huggingface/diffusers.git
```

### Set up kolors

HF model: https://huggingface.co/Kwai-Kolors/Kolors-diffusers

HF pipeline: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/kolors.md


## Run

### Run 1024*1024 without compile (the original pytorch HF diffusers baseline)
```
python3 onediff_diffusers_extensions/examples/kolors/text_to_image_kolors.py \
--saved-image kolors.png
```

### Run 1024*1024 with compile [oneflow backend]

```
python3 onediff_diffusers_extensions/examples/kolors/text_to_image_kolors.py \
--compiler oneflow \
--saved-image kolors_oneflow_compile.png
```

### Run 1024*1024 with compile [nexfort backend]

```
python3 onediff_diffusers_extensions/examples/kolors/text_to_image_kolors.py \
--compiler nexfort \
--compiler-config '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last"}' \
--saved-image kolors_nexfort_compile.png
```

## Performance comparation

**Testing on an NVIDIA RTX 4090 GPU, using a resolution of 1024x1024 and 50 steps:**

Data update date: 2024-07-23

| Framework          | Iteration Speed (it/s) | E2E Time (seconds) | Max Memory Used (GiB) | Warmup time (seconds) <sup>1</sup> | Warmup with Cache time (seconds) |
|--------------------|------------------------|--------------------|-----------------------|-------------|------------------------|
| PyTorch            | 8.11                   | 6.55               | 20.623                | 7.09        | -                      |
| OneDiff (OneFlow)  | 15.16 (+86.9%)                 | 3.86 (-41.1%)              | 20.622                | 39.61       | 7.47                   |
| OneDiff (NexFort)  | 14.68 (+81.0%)                 | 3.71 (-43.4%)              | 21.623                | 190.14      | 50.46                  |

 <sup>1</sup> OneDiff Warmup with Compilation time is tested on AMD EPYC 7543 32-Core Processor.

**Testing on NVIDIA A100-PCIE-40GB:**

Data update date: 2024-07-23

| Framework          | Iteration Speed (it/s) | E2E Time (seconds) | Max Memory Used (GiB) | Warmup time (seconds) <sup>2</sup> | Warmup with Cache time (seconds) |
|--------------------|------------------------|--------------------|-----------------------|-------------|------------------------|
| PyTorch            | 8.36                   | 6.34               | 20.622                | 7.88        | -                      |
| OneDiff (OneFlow)  | 11.54 (+38.0%)                 | 4.69 (-26.0%)              | 20.627                | 50.02       | 12.82                  |
| OneDiff (NexFort)  | 10.53 (+26.0%)                 | 5.02 (-20.8%)              | 21.622                | 269.89      | 73.31                  |

 <sup>2</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz.

**Testing on NVIDIA A100-SXM4-80GB:**

Data update date: 2024-07-23

| Framework          | Iteration Speed (it/s) | E2E Time (seconds) | Max Memory Used (GiB) | Warmup time (seconds) <sup>3</sup> | Warmup with Cache time (seconds) |
|--------------------|------------------------|--------------------|-----------------------|-------------|------------------------|
| PyTorch            | 9.88                   | 5.38               | 20.622                | 6.61        | -                      |
| OneDiff (OneFlow)  | 13.70 (+38.7%)                 | 3.96 (-26.4%)              | 20.627                | 52.93       | 11.79                  |
| OneDiff (NexFort)  | 13.20 (+33.6%)                 | 4.04 (-24.9%)              | 21.622                | 150.78      | 58.07                  |

 <sup>3</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Platinum 8468.

## Dynamic shape.

Run:

```
# oneflow
python3 onediff_diffusers_extensions/examples/kolors/text_to_image_kolors.py \
--compiler oneflow \
--run_multiple_resolutions 1 \
--saved-image kolors_oneflow_compile.png
```

or

```
# nexfort
python3 onediff_diffusers_extensions/examples/kolors/text_to_image_kolors.py \
--height 512 \
--width 768 \
--compiler nexfort \
--compiler-config '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last", "dynamic": true}' \
--run_multiple_resolutions 1 \
--saved-image kolors_nexfort_compile.png
```

## Quality

The quality report for accelerating the kolors model with onediff is located at:
https://github.com/siliconflow/odeval/tree/main/models/kolors
