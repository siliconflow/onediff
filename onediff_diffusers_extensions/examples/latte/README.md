# Run Latte with nexfort backend(Beta Release)


1. [Environment Setup](#environment-setup)
   - [Set up onediff](#set-up-onediff)
   - [Set up nexfort backend](#set-up-nexfort-backend)
   - [Set up Latte](#set-up-latte)
2. [Run](#run)
   - [Run 1024*1024 without compile](#run-10241024-without-compile)
   - [Run 1024*1024 with compile](#run-10241024-with-compile)
3. [Performance Comparison](#performance-comparison)
4. [Quality](#quality)

## Environment setup
### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

### Set up Latte
HF model: https://huggingface.co/maxin-cn/Latte-1

GIthub source: https://github.com/Vchitect/Latte

## Run
model_id_or_path_to_latte is the model id or model path of latte, such as `maxin-cn/Latte-1` or `/data/hf_models/Latte-1/`

### Go to the onediff folder
```
cd onediff
```

### Run 1024*1024 without compile(the original pytorch HF diffusers pipeline)
```
python3 ./benchmarks/text_to_video_latte.py \
--model maxin-cn/Latte-1 \
--steps 50 \
--compiler none \
----output-video ./latte.mp4 \
--prompt "a cat wearing sunglasses and working as a lifeguard at pool."
```

### Run 1024*1024 with compile
```
python3 ./benchmarks/text_to_video_latte.py \
--model maxin-cn/Latte-1 \
--steps 50 \
--compiler nexfort \
----output-video ./latte_compile.mp4 \
--prompt "a cat wearing sunglasses and working as a lifeguard at pool."
```

## Performance Comparison

### Metric

#### On A100
| Metric                                           | NVIDIA A100-PCIE-40GB (1024 * 1024) |
| ------------------------------------------------ | ----------------------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-06-19                          |
| PyTorch iteration speed                          | 1.60it/s                            |
| OneDiff iteration speed                          | 2.27it/s(+41.9%)                    |
| PyTorch E2E time                                 | 32.618s                             |
| OneDiff E2E time                                 | 22.601s(-30.7%)                     |
| PyTorch Max Mem Used                             | 28.208GiB                           |
| OneDiff Max Mem Used                             | 24.753GiB                           |
| PyTorch Warmup with Run time                     | 33.291s                             |
| OneDiff Warmup with Compilation time<sup>1</sup> | 572.877s                            |
| OneDiff Warmup with Cache time                   | 148.068s                            |

 <sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz. Note this is just for reference, and it varies a lot on different CPU.

#### nexfort compile config and warmup cost
- compiler-config 
  - setting `--compiler-config '{"mode": "max-optimize:max-autotune:freezing:benchmark:low-precision", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": false, "triton.fuse_attention_allow_fp16_reduction": false}}` will help to make the best performance but the compilation time is about 572 seconds
  - setting `--compiler-config '{"mode": "max-autotune", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": false, "triton.fuse_attention_allow_fp16_reduction": false}}` will reduce compilation time to about 236 seconds and just slightly reduce the performance
- fuse_qkv_projections: True

## Quality

When using nexfort as the backend for onediff compilation acceleration (right video), the generated video are lossless.

<p align="center">
<img src="../../../imgs/latte_nexfort.gif">
</p>
