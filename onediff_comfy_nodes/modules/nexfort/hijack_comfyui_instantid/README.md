## Accelerating ComfyUI_InstantID with OneDiff
### Environment
Please Refer to the Readme in the Respective Repositories for Installation Instructions.
#### Install OneDiff

When you have completed these steps, follow the [instructions](https://github.com/siliconflow/onediff/blob/ba93c5a68607abefd38ffed9e6a17bed48c01a81/README.md?plain=1#L224) to install OneDiff.
Then follow the [guide](https://github.com/siliconflow/onediff/blob/0819aa41c8a910add96400265f3165f9d8d3634c/onediff_comfy_nodes/README.md?plain=1#L86) to install ComfyUI OneDiff extension

#### Install ComfyUI

```
cd ComfyUI/custom_nodes
git clone https://github.com/comfyanonymous/ComfyUI
git reset --hard 2d4164271634476627aae31fbec251ca748a0ae0
```
When you have completed these steps, follow the [instructions](https://github.com/comfyanonymous/ComfyUI) to install ComfyUI

#### Install ComfyUI_InstantID

```
cd ComfyUI/custom_nodes
git clone https://github.com/cubiq/ComfyUI_InstantID.git
git reset --hard d8c70a0cd8ce0d4d62e78653674320c9c3084ec1
```
When you have completed these steps,follow the [instructions](https://github.com/cubiq/ComfyUI_InstantID) to install ComfyUI_InstantID

### Quick Start

> Recommend running the official example of ComfyUI_InstantID now, and then trying OneDiff acceleration.
> You can Load these images in ComfyUI to get the full workflow.

Experiment (GeForce RTX 3090) Workflow for OneDiff Acceleration in ComfyUI_InstantID:

1. Replace the **`Load Checkpoint`** node with **`Load Checkpoint - OneDiff`** node.
2. Add a **`Batch Size Patcher`** node before the **`Ksampler`** node (due to temporary lack of support for dynamic batch size).
As follows:
![workflow (20)](https://github.com/siliconflow/onediff/assets/117806079/492a83a8-1a5b-4fb3-9e53-6d53e881a3f8)

Note that you can download all images in this page and then drag or load them on ComfyUI to get the workflow embedded in the image.
![oneflow_basic](https://github.com/siliconflow/oneflow/assets/117806079/81016bd8-3ec8-457f-850f-9c486bfd2d0c)


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


### InstantID_basic
#### WorkFlow Description
source: https://github.com/cubiq/ComfyUI_InstantID/blob/main/examples/InstantID_basic.json
| InstantID | Baseline (non-optimized)                                                                                         | OneDiff (optimized)                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| WorkFlow          |![InstantID_basic_torch](https://github.com/siliconflow/sd-team/assets/117806079/d649539c-7e8e-449f-b7b5-08622e6f93cc) |![InstantID_basic_oneflow](https://github.com/siliconflow/sd-team/assets/117806079/c752ca4b-7d81-49b4-915a-9c3088227e9d)|

#### Performance Comparison

Timings for 30 steps at 1024*1024

| Accelerator           | Baseline (non-optimized) | OneDiff (optimized) | Percentage improvement |
| --------------------- | ------------------------ | ------------------- | ---------------------- |
| GeForce RTX 3090 |  12.69 s                   | 9 s              |    29.1 %          |

### InstantID_IPAdapter
#### WorkFlow Description
source: https://github.com/cubiq/ComfyUI_InstantID/blob/main/examples/InstantID_IPAdapter.json

| InstantID_IPAdapter | Baseline (non-optimized)                                                                                         | OneDiff (optimized)                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| WorkFlow          |![InstantID_IPAdapter_torch](https://github.com/siliconflow/sd-team/assets/117806079/ba4ba6a9-f9d8-4921-85dd-be00c72f20a6) | ![InstantID_IPAdapter_oneflow](https://github.com/siliconflow/sd-team/assets/117806079/46533f74-7634-4839-8c3e-c555c78eca63) |

#### Performance Comparison

Timings for 30 steps at 1024*1024

| Accelerator           | Baseline (non-optimized) | OneDiff (optimized) | Percentage improvement |
| --------------------- | ------------------------ | ------------------- | ---------------------- |
| GeForce RTX 3090 |   13.23 s                   |  9.33 s              |   29.5%                |

- **Note:**
   - Consider setting `ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION=0` and `ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION=0` to ensure computational precision, but expect a potential 5% reduction in performance.

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.
