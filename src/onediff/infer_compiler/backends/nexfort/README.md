## nexfort backend for compiler in onediff
###  Dependency
```
pip3 install --pre -U torch==2.4.0.dev20240507 torchaudio==2.2.0.dev20240507+cu124 torchvision==0.19.0.dev20240507+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
pip3 install -U torchao==0.1
```
It also supports the stable versions of `torch2.3.0`, `torch2.2.0`, and `torch2.1.0`.

### Install nexfort

Before installing nextfort, please make sure that the corresponding PyTorch and CUDA environments are installed.

```
pip3 install nexfort
```

### Run pixart alpha (with nexfort backend)

```
# model_id_or_path_to_PixArt-XL-2-1024-MS: /data/hf_models/PixArt-XL-2-1024-MS/ 
python3 ./benchmarks/text_to_image.py --model model_id_or_path_to_PixArt-XL-2-1024-MS --scheduler none --steps 20 --compiler nexfort --output-image ./pixart_alpha_nex.png
```
Performance on NVIDIA A100-PCIE-40GB:
- Warmup time: 771.418s
- Inference time: 2.045s
- Iterations per second: 10.743
- Max used CUDA memory: 13.855GiB

### Compilation cache speeds up recompilation

Setting cache:
```
# Enabled Inductor - FX Graph Cache. Default Off
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

# Setting Inductor - Autotuning Cache DIR. This cache is enabled by default.
export TORCHINDUCTOR_CACHE_DIR=~/.torchinductor
```

Clear Cache:
```
python3 -m nexfort.utils.clear_inductor_cache
```

Advanced cache functionality is currently in development.

### Dynamic shape
Onediff's nexfort backend also supports out-of-the-box dynamic shape inference. You just need to enable `dynamic` during compilation, as in `'{"mode": "max-autotune
", "dynamic": true}'`. To understand how dynamic shape support works, please refer to the <https://pytorch.org/docs/stable/generated/torch.compile.html> and <https://github.com/pytorch/pytorch/blob/main/docs/source/torch.compiler_dynamic_shapes.rst> page. To avoid over-specialization and re-compilation, you need to initially call your model with a non-typical shape. For example: you can first call your Stable Diffusion model with a shape of 512x768 (height != width).

Test SDXL:
```
# The best practice mode configuration for dynamic shape is `max-optimize:max-autotune:freezing`.
python3 ./onediff_diffusers_extensions/examples/text_to_image_sdxl.py \
--height 512 \
--width 768 \
--compiler nexfort \
--compiler-config '{"mode": "max-optimize:max-autotune:freezing", "memory_format": "channels_last", "dynamic": true}' \
--run_multiple_resolutions 1 \
--run_rare_resolutions 1
```

Test PixArt alpha:
```
python3 ./benchmarks/text_to_image.py \
--model /data/hf_models/PixArt-XL-2-1024-MS/ \
--scheduler none \
--steps 20 \
--height 512 \
--width 768 \
--compiler nexfort \
--compiler-config '{"mode": "max-optimize:max-autotune:freezing", "memory_format": "channels_last", "dynamic": true}' \
--run_multiple_resolutions 1
```
