# Run PixArt with nexfort backend (Beta Release)


1. [Environment Setup](#environment-setup)
   - [Set up onediff](#set-up-onediff)
   - [Set up nexfort backend](#set-up-nexfort-backend)
   - [Set up PixArt](#set-up-pixart)
2. [Run](#run)
   - [Run 1024*1024 without compile](#run-10241024-without-compile)
   - [Run 1024*1024 with compile](#run-10241024-with-compile)
3. [Performance Comparison](#performance-comparison)
4. [Quantization](#quantization)
   - [Run](#run-1)
   - [Metric](#metric-1)
   - [Optimizing Quantization Precision](#optimizing-quantization-precision)
5. [Quality](#quality)

## Environment setup
### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

### Set up PixArt


HF model:

 - PixArt-alpha: https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS
 - PixArt-sigma: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS

HF pipeline: https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart

## Run

model_id_or_path_to_PixArt-XL-2-1024-MS is the model id or model path of pixart alpha, such as `PixArt-alpha/PixArt-XL-2-1024-MS` or `/data/hf_models/PixArt-XL-2-1024-MS/`

> [!NOTE]
Compared to PixArt-alpha, PixArt-sigma extends the token length of the text encoder and introduces a new attention module capable of compressing key and value tokens, yet it still maintains consistency in the model architecture. The nexfort backend of onediff can support compilation acceleration for any version of PixArt.

### Go to the onediff folder
```
cd onediff
```

### Run 1024*1024 without compile (the original pytorch HF diffusers pipeline)
```
# To test sigma, specify the --model parameter as `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`.
python3 ./benchmarks/text_to_image.py \
--model PixArt-alpha/PixArt-XL-2-1024-MS \
--scheduler none \
--steps 20 \
--compiler none \
--output-image ./pixart_alpha.png \
--prompt "product photography, world of warcraft orc warrior, white background"
```

### Run 1024*1024 with oneflow backend compile

```
python3 ./benchmarks/text_to_image.py \
--model PixArt-alpha/PixArt-XL-2-1024-MS \
--scheduler none \
--steps 20 \
--compiler oneflow \
--output-image ./pixart_alpha_compile.png \
--prompt "product photography, world of warcraft orc warrior, white background"
```

### Run 1024*1024 with nexfort backend compile
```
python3 ./benchmarks/text_to_image.py \
--model PixArt-alpha/PixArt-XL-2-1024-MS \
--scheduler none \
--steps 20 \
--compiler nexfort \
--output-image ./pixart_alpha_compile.png \
--prompt "product photography, world of warcraft orc warrior, white background"
```

## Performance Comparison

### Metric

#### On 4090
| Metric                                           |NVIDIA GeForce RTX 4090 (1024 * 1024)|
| ------------------------------------------------ | ----------------------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-07-06                          |
| PyTorch iteration speed                          | 7.591it/s                           |
| OneDiff iteration speed                          | 14.308it/s(+88.5%)                  |
| PyTorch E2E time                                 | 2.881s                              |
| OneDiff E2E time                                 | 1.509s(-47.6%)                      |
| PyTorch Max Mem Used                             | 14.447GiB                           |
| OneDiff Max Mem Used                             | 13.571GiB                           |
| PyTorch Warmup with Run time                     | 3.314s                              |
| OneDiff Warmup with Compilation time<sup>1</sup> | 244.500s                            |
| OneDiff Warmup with Cache time                   | 80.866s                             |

 <sup>1</sup> OneDiff warmup with compilation time is tested on AMD EPYC 7543 32-Core Processor. Note this is just for reference, and it varies a lot on different CPU.

#### On A100
| Metric                                           | NVIDIA A100-PCIE-40GB (1024 * 1024) |
| ------------------------------------------------ | ----------------------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-05-23                          |
| PyTorch iteration speed                          | 8.623it/s                           |
| OneDiff iteration speed                          | 10.743it/s(+24.6%)                  |
| PyTorch E2E time                                 | 2.568s                              |
| OneDiff E2E time                                 | 1.992s(-22.4%)                      |
| PyTorch Max Mem Used                             | 14.445GiB                           |
| OneDiff Max Mem Used                             | 13.855GiB                           |
| PyTorch Warmup with Run time                     | 4.100s                              |
| OneDiff Warmup with Compilation time<sup>2</sup> | 510.170s                            |
| OneDiff Warmup with Cache time                   | 111.563s                            |

 <sup>2</sup> Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz.

#### On H800
| Metric                                           |  NVIDIA H800-NVL-80GB (1024 * 1024) |
| ------------------------------------------------ | ----------------------------------- |
| Data update date(yyyy-mm-dd)                     | 2024-05-29                          |
| PyTorch iteration speed                          | 21.067it/s                          |
| OneDiff iteration speed                          | 24.518it/s(+16.4%)                  |
| PyTorch E2E time                                 | 1.102s                              |
| OneDiff E2E time                                 | 0.865s(-21.5%)                      |
| PyTorch Max Mem Used                             | 14.468GiB                           |
| OneDiff Max Mem Used                             | 13.970GiB                           |
| PyTorch Warmup with Run time                     | 1.741s                              |
| OneDiff Warmup with Compilation time<sup>3</sup> | 718.539s                            |
| OneDiff Warmup with Cache time                   | 131.776s                            |

 <sup>3</sup> Intel(R) Xeon(R) Platinum 8468.

#### The nexfort backend compile config and warmup cost

- compiler-config
  - default is `{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last"}` in `/benchmarks/text_to_image.py`. This mode supports dynamic shapes.
  - setting `--compiler-config '{"mode": "max-autotune", "memory_format": "channels_last"}'` will reduce compilation time and just slightly reduce the performance.
  - setting `--compiler-config '{"mode": "max-optimize:max-autotune:freezing:benchmark:low-precision:cudagraphs", "memory_format": "channels_last"}'` will help achieve the best performance, but it increases the compilation time and affects stability.
  - setting `--compiler-config '{"mode": "jit:disable-runtime-fusion", "memory_format": "channels_last"}'` will reduce compilation time to 20 seconds, but will reduce the performance.
- fuse_qkv_projections: True

## Quantization

> [!NOTE]
Quantization is a feature for onediff enterprise.

Onediff's nexfort backend works closely with Torchao to support model quantization. Quant can reduce the runtime memory requirement and increase the inference speed.

### Run

```
python3 ./benchmarks/text_to_image.py \
--model PixArt-alpha/PixArt-XL-2-1024-MS \
--scheduler none \
--steps 20 \
--output-image ./pixart_alpha_fp8.png \
--prompt "product photography, world of warcraft orc warrior, white background" \
--compiler nexfort \
--compiler-config '{"mode": "quant:max-optimize:max-autotune:freezing:benchmark:low-precision:cudagraphs", "memory_format": "channels_last"}' \
--quantize \
--quantize-config '{"quant_type": "fp8_e4m3_e4m3_dynamic"}'
```

**Optimal balance**. Currently, multiple quant types are supported, such as `int8_dynamic`, `fp8_e4m3_e4m3_weightonly`, `fp8_e4m3_e4m3_dynamic_per_tensor`, `fp8_e4m3_e4m3_weightonly_per_tensor`. Additionally, the format of activations and weights supports switching between `e4m3` and `e5m2`, such as `fp8_e4m3_e5m2_dynamic`.

It is recommended to try different types of quantization to find the optimal situation in terms of performance and quality.

### Metric

NVIDIA H800 (1024 * 1024)

| quant_type                       | E2E Inference Time | Iteration speed    | Max Used CUDA Memory |
|----------------------------------|--------------------|--------------------|----------------------|
| fp8_e4m3_e4m3_dynamic            | 0.824s (-25.2%)    | 25.649 (+21.8%)   | 13.400 GiB           |
| fp8_e4m3_e4m3_dynamic_per_tensor | 0.834s (-24.3%)    | 25.323 (+20.2%)   | 13.396 GiB           |
| int8_dynamic                     | 0.832s (-24.5%)    | 25.391 (+20.5%)   | 13.367 GiB           |

### Optimizing Quantization Precision

Quantization of the model's layers can be selectively performed based on precision. Download `fp8_e4m3.json` or `per_tensor_fp8_e4m3.json` from https://huggingface.co/siliconflow/PixArt-alpha-onediff-nexfort-fp8.

Run:
```
python3 ./benchmarks/text_to_image.py \
--model PixArt-alpha/PixArt-XL-2-1024-MS \
--scheduler none \
--steps 20 \
--output-image ./pixart_alpha_fp8_90%.png \
--prompt "product photography, world of warcraft orc warrior, white background" \
--compiler nexfort \
--compiler-config '{"mode": "quant:max-optimize:max-autotune:freezing:benchmark:low-precision:cudagraphs", "memory_format": "channels_last"}' \
--quantize \
--quantize-config '{"quant_type": "fp8_e4m3_e4m3_dynamic"}' \
--quant-submodules-config-path /path/to/fp8_e4m3.json
```

## Quality

When using nexfort as the backend for onediff compilation acceleration, the generated images are lossless. With nexfort's quantized versions (fp8 or int8), there are only slight differences.

<p align="center">
<img src="../../../imgs/nexfort_sample_quality.png">
</p>
