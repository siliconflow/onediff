# Run SD3 with nexfort backend (Beta Release)

## Environment setup
### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

### Set up SD3
HF model: https://huggingface.co/stabilityai/stable-diffusion-3-medium

HF pipeline: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_3.md

## Run

### Run 1024*1024 without compile (the original pytorch HF diffusers baseline)
```
python3 onediff_diffusers_extensions/examples/sd3/text_to_image_sd3.py \
    --saved-image sd3.png
```

### Run 1024*1024 with compile

```
python3 onediff_diffusers_extensions/examples/sd3/text_to_image_sd3.py \
    --compiler-config '{"mode": "quant:max-optimize:max-autotune:freezing:benchmark:low-precision:cudagraphs", \
    "memory_format": "channels_last"}' \
    --saved-image sd3_compile.png
```

## Performance comparation

Testing on H800, with image size of 1024*1024, iterating 28 steps.

|                | Iteration speed     | E2E Inference Time          | Max CUDA Memory Used |
|----------------|---------------------|-----------------------------|----------------------|
| Baseline       |     15.56 it/s      |          1.96 s             | 18.784 GiB           |
| Nexfort compile| 25.91 it/s (+66.52%)|       1.15 s (-41.33%)      | 18.324 GiB           |


## Quality
TODO
