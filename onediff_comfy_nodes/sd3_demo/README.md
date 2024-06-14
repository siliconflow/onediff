## Accelerate SD3 by using onediff
huggingface: https://huggingface.co/stabilityai/stable-diffusion-3-medium 


### Feature
- ✅ Multiple resolutions

### Performance

- Timings for 28 steps at 1024x1024
- OneDiff[Nexfort] Compile mode: max-optimize:max-autotune:low-precision

| Accelerator           | Baseline (non-optimized) | OneDiff (optimized) | Percentage improvement |
| --------------------- | ------------------------ | ------------------- | ---------------------- |
| NVIDIA A800-SXM4-80GB | ~4.03 sec                | ~2.93 sec           | ~27.29 %               |




The following table shows the comparison of the plot, seed=1, Baseline (non optimized) on the left, and OneDiff (optimized) on the right

|                                                                                                                      |                                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| ![sd3_baseline_00001_](https://github.com/siliconflow/onediff/assets/109639975/c86f2dc8-fc6f-4cc7-b85d-d4d973594ee6) | ![sd3_speedup_00001_](https://github.com/siliconflow/onediff/assets/109639975/c81b3fc9-d588-4ba1-9911-ae3a8a8d2454) |


### Multiple resolutions
test with multiple resolutions and support shape switching in a single line of Python code
```
[print(f"Testing resolution: {h}x{w}") for h in [1024, 512, 768, 256] for w in [1024, 512, 768, 256]]
```

## Usage Example

### Install

```shell
# python 3.10 
COMFYUI_DIR=$pwd/ComfyUI
git clone https://github.com/siliconflow/onediff.git 
cd onediff && pip install -r onediff_comfy_nodes/sd3_demo/requirements.txt && pip install -e .
ln -s $pwd/onediff/onediff_comfy_nodes  $COMFYUI_DIR/custom_nodes
git clone https://github.com/comfyanonymous/ComfyUI.git
```

<details close>
<summary> test_install.py </summary>

```python
# Compile arbitrary models (torch.nn.Module)
import torch
from onediff.utils.import_utils import is_nexfort_available
assert is_nexfort_available() == True

import onediff.infer_compiler as infer_compiler

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

mod = MyModule().to("cuda").half()
with torch.inference_mode():
    compiled_mod = infer_compiler.compile(mod,
        backend="nexfort",
        options={"mode": "max-autotune:cudagraphs", "dynamic": True, "fullgraph": True},
    )
    print(compiled_mod(torch.randn(10, 100, device="cuda").half()).shape)
    
print("Successfully installed～")
```
</details>

### Run ComfyUI
```shell
# run comfyui
# For CUDA Graph
export NEXFORT_FX_CUDAGRAPHS=1
# For best performance
export TORCHINDUCTOR_MAX_AUTOTUNE=1
# Enable CUDNN benchmark
export NEXFORT_FX_CONV_BENCHMARK=1
# Faster float32 matmul
export NEXFORT_FX_MATMUL_ALLOW_TF32=1
# For graph cache to speedup compilation
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
# For persistent cache dir
export TORCHINDUCTOR_CACHE_DIR=~/.torchinductor_cache
cd $COMFYUI_DIR && python main.py --gpu-only --disable-cuda-malloc
```

### WorkFlow
![WorkFlow](https://github.com/siliconflow/onediff/assets/109639975/a385fac5-1f82-4905-a941-4c71ff1c616e)

