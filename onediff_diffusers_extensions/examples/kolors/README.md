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
pip install git+https://github.com/asomoza/diffusers.git@add-kolors-support
```

### Set up kolors

HF model: https://huggingface.co/Kwai-Kolors/Kolors-diffusers

HF pipeline: https://github.com/huggingface/diffusers/blob/3d1c702912f62fd3489a2e61ecf67c10e4419abd/docs/source/en/api/pipelines/kolors.md


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


## Quality

<p align="center">
<img src="../../../imgs/kolors_demo.png">
</p>
