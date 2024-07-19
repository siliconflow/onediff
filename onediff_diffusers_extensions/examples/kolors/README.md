# Run kolors with onediff (Beta Release)


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

Testing on an NVIDIA RTX 4090 GPU, using a resolution of 1024x1024 and 50 steps:

| Framework | Iteration Speed (it/s) | E2E Time (seconds) | Max Memory Used (GiB) |
|-----------|------------------------|--------------------|-----------------------|
| PyTorch   | 8.11                   | 6.55               | 20.623                |
| OneDiff (OneFlow) | 15.16 (+86.9%)         | 3.86 (-41.1%)               | 20.622                |
| OneDiff (NexFort) | 14.68 (+81.0%)         | 3.71 (-43.4%)               | 21.623                |

Testing on NVIDIA A100-PCIE-40GB:

| Framework | Iteration Speed (it/s) | E2E Time (seconds) | Max Memory Used (GiB) |
|-----------|------------------------|--------------------|-----------------------|
| PyTorch   | 8.36                   | 6.34               | 20.622                |
| OneDiff (OneFlow) | 11.54 (+38.0%)         | 4.69 (-26.0%)               | 20.627                |
| OneDiff (NexFort) | 10.53 (+26.0%)         | 5.02 (-20.8%)               | 21.622                |

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

<p align="center">
<img src="../../../imgs/kolors_demo.png">
</p>
