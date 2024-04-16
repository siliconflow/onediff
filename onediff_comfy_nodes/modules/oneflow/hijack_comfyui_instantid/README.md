## Accelerating ComfyUI_InstantID with OneDiff
### Environment
Please Refer to the Readme in the Respective Repositories for Installation Instructions.

- ComfyUI:
  - github: https://github.com/comfyanonymous/ComfyUI
  - commit: 4bd7d55b9028d79829a645edfe8259f7b7a049c0 
  - Date:   Thu Apr 11 22:43:05 2024 -0400
  
- ComfyUI_InstantID:
  - github: https://github.com/cubiq/ComfyUI_InstantID
  - commit: e9cc7597b2a7cd441065418a975a2de4aa2450df 
  - Date:   Tue Apr 9 14:05:15 2024 +0200
  
- OneDiff:
  - github: https://github.com/siliconflow/onediff 

### Quick Start

> Recommend running the official example of ComfyUI_InstantID now, and then trying OneDiff acceleration. 
> You can Load these images in ComfyUI to get the full workflow.

Experiment (NVIDIA A100-PCIE-40GB) Workflow for OneDiff Acceleration in ComfyUI_InstantID:

1. Replace the **`Load Checkpoint`** node with **`Load Checkpoint - OneDiff`** node. 
2. Add a **`Batch Size Patcher`** node before the **`Ksampler`** node (due to temporary lack of support for dynamic batch size).
As follows:

Note that you can download all images in this page and then drag or load them on ComfyUI to get the workflow embedded in the image.
![oneflow_basic](https://github.com/siliconflow/onediff/assets/109639975/5ab47746-04e8-4753-afcf-7d95161f542e)


<details close>
<summary> Download the required model files </summary>

InstantID requires `insightface`, you need to add it to your libraries together with `onnxruntime` and `onnxruntime-gpu`.

The InsightFace model is **antelopev2** (not the classic buffalo_l). Download the models (for example from [here](https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing) or [here](https://huggingface.co/MonsterMMORPG/tools/tree/main)), unzip and place them in the `ComfyUI/models/insightface/models/antelopev2` directory.


##### For NA/EU users
```shell
cd ComfyUI
# Load Checkpoint
wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# Load InstantID Model
mkdir -p models/instantid
wget -O models/instantid/ip-adapter.bin https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin


# Load ControlNet Model
wget -O models/controlnet/diffusion_pytorch_model.safetensors https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors

```

##### For CN users
```shell
cd ComfyUI
wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# Load InstantID Model
mkdir -p models/instantid
wget -O models/instantid/ip-adapter.bin https://hf-mirror.com/InstantX/InstantID/resolve/main/ip-adapter.bin


# Load ControlNet Model
wget -O models/controlnet/diffusion_pytorch_model.safetensors https://hf-mirror.com/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors
```

</details>


- **Note:**
   - Consider setting `ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION=0` and `ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION=0` to ensure computational precision, but expect a potential 5% reduction in performance.





