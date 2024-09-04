# OneDiff compiler for inference

## With nexfort compiler backend(Beta release)
OneDiff Nexfort is a lightweight [torch 2.0 compiler backend](https://pytorch.org/docs/stable/torch.compiler.html) specially optimized for Diffusion Models.

Currently, it is especially for DiT(Diffusion Transformer) models which is the backbone of [SD3](https://stability.ai/news/stable-diffusion-3) and [Sora](https://openai.com/sora/).

### Installation
####  Dependency
```
pip3 install -U torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 torchao==0.1
```
The current version of nexfort is compatible with `torch 2.3.0` and `torch 2.4.0`.

#### Install nexfort
Reference: Install nexfort: https://github.com/siliconflow/onediff?tab=readme-ov-file#nexfort

#### Install onediff
Reference: https://github.com/siliconflow/onediff?tab=readme-ov-file#3-install-onediff

### Usage
```python
from onediff.infer_compiler import compile

# module is the model you want to compile
options = {"mode": "O3"}  # mode can be O2 or O3
compiled = compile(module, backend="nexfort", options=options)
```

If you are using [onediffx for HF diffusers, you can use compile_pipe](https://github.com/siliconflow/onediff/tree/main/onediff_diffusers_extensions#compile-save-and-load-pipeline) like this:
```
from onediffx import compile_pipe
options = '{"mode": "O3", "memory_format": "channels_last"}'
pipe = compile_pipe(pipe, backend="nexfort", options=options, fuse_qkv_projections=True)
```


### Available Compiler Modes
Config Example:
```
options = {"mode": "max-optimize:max-autotune:low-precision:cache-all"}
```

| Mode | Description |
| - | - |
| `cache-all` | Cache all the compiled stuff to speed up reloading and recompiling. |
| `max-autotune` | Enable all the kernel autotuning options to find out the best kernels, this might slow down the compilation. |
| `max-optimize` | Enable the ***most*** extreme optimization strategies like the most aggressive fusion kernels to maximize the performance, this might slow down the compilation and require long autotuning. |
| `cudagraphs` | Enable CUDA Graphs to reduce CPU overhead. |
| `freezing` | Freezing will attempt to inline weights as constants in optimization and run constant folding and other optimizations on them. After freezing, weights can no longer be updated. |
| `low-precision` | Enable low precision mode. This will allow some math computations happen in low precision to speed up the overall performance. |

### Suggested Combination Modes
Config Example:
```
options = {"mode": "O3"}
```

| Combination | Description |
| - | - |
| `O2` | This is the most suggested combination of compiler modes. This mode requires support for most models, ensuring model accuracy, and supporting dynamic resolution. |
| `O3` | This aims for efficiency. |

`O2` and `O3` are approximately equal to `options = {"mode": "max-optimize:max-autotune:low-precision:cache-all"}`, but `O2` has higher precision.

### Run pixart alpha (with nexfort backend)

Details at: https://github.com/siliconflow/onediff/tree/main/onediff_diffusers_extensions/examples/pixart_alpha

### Compilation cache to speed up recompilation

Setting cache:
```
# Enabled Cache Compiled Graph. Default Off
export NEXFORT_GRAPH_CACHE=1

# Setting Inductor - Autotuning Cache DIR. This cache is enabled by default.
export TORCHINDUCTOR_CACHE_DIR=~/torchinductor
```

Clear Cache:
```
python3 -m nexfort.utils.clear_inductor_cache
```

Advanced cache functionality is currently in development.

### Dynamic shape
Onediff's nexfort backend also supports out-of-the-box dynamic shape inference. You just need to enable `dynamic` during compilation, as in `'{"mode": "max-autotune", "dynamic": true}'`. To understand how dynamic shape support works, please refer to the <https://pytorch.org/docs/stable/generated/torch.compile.html> and <https://github.com/pytorch/pytorch/blob/main/docs/source/torch.compiler_dynamic_shapes.rst> page. To avoid over-specialization and re-compilation, you need to initially call your model with a non-typical shape. For example: you can first call your Stable Diffusion model with a shape of 512x768 (height != width).

Test SDXL:
```
# The best practice mode configuration for dynamic shape is `max-optimize:max-autotune:low-precision`.
python3 ./onediff_diffusers_extensions/examples/text_to_image_sdxl.py \
--height 512 \
--width 768 \
--compiler nexfort \
--compiler-config '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last", "dynamic": true}' \
--run_multiple_resolutions 1 \
--run_rare_resolutions 1
```

Test PixArt alpha:
```
python3 ./benchmarks/text_to_image.py \
--model PixArt-alpha/PixArt-XL-2-1024-MS \
--scheduler none \
--steps 20 \
--height 512 \
--width 768 \
--compiler nexfort \
--compiler-config '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last", "dynamic": true}' \
--run_multiple_resolutions 1
```
