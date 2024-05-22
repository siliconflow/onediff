## Accelerating ComfyUI_IPAdapter_plus with OneDiff
### Environment
Please Refer to the Readme in the Respective Repositories for Installation Instructions.
#### Install OneDiff
```
git clone https://github.com/siliconflow/oneflow.git
```
When you have completed these steps, follow the [instructions](https://github.com/siliconflow/onediff) to install OneDiff.
Then follow the [guide](https://github.com/siliconflow/onediff/blob/0819aa41c8a910add96400265f3165f9d8d3634c/onediff_comfy_nodes/README.md?plain=1#L86) to install ComfyUI OneDiff extension


#### Install ComfyUI

```
git clone https://github.com/comfyanonymous/ComfyUI.git
git reset --hard  4bd7d55b9028d79829a645edfe8259f7b7a049c0
```
When you have completed these steps, follow the [instructions](https://github.com/comfyanonymous/ComfyUI) to install ComfyUI

#### Install ComfyUI_IPAdapter_plus

```
cd ComfyUI/custom_nodes
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
git reset --hard  417d806e7a2153c98613e86407c1941b2b348e88
```
When you have completed these steps, follow the [instructions](https://github.com/cubiq/ComfyUI_IPAdapter_plus) instructions for installation

#### Install ComfyUI_InstantID

```
cd ComfyUI/custom_nodes
git clone https://github.com/cubiq/ComfyUI_InstantID.git
git reset --hard  e9cc7597b2a7cd441065418a975a2de4aa2450df
```
When you have completed these steps,follow the [instructions](https://github.com/cubiq/ComfyUI_InstantID) below to install ComfyUI_InstantID

#### Install ComfyUI-AnimateDiff-Evolved

```
cd ComfyUI/custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
git reset --hard  f9e0343f4c4606ee6365a9af4a7e16118f1c45e1
```
When you have completed these steps, follow the [instructions](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved/)  for installation


### Quick Start

> Recommend running the official example of ComfyUI_IPAdapter_plus now, and then trying OneDiff acceleration. 

Experiment (GeForce RTX 3090) Workflow for OneDiff Acceleration in ComfyUI_IPAdapter_plus:

1. Replace the **`Load Checkpoint`** node with **`Load Checkpoint - OneDiff`** node. 
2. Add a **`Batch Size Patcher`** node before the **`Ksampler`** node (due to temporary lack of support for dynamic batch size).
As follows:

### ipadapter_advanced_basic
#### WorkFlow Description
source: https://github.com/cubiq/ComfyUI_IPAdapter_plus/blob/main/examples/ipadapter_advanced.json

| ipadapter_advanced_basic | Baseline (non-optimized)                                                                                         | OneDiff (optimized)                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| WorkFlow          |![ipadapter_advanced_torch](https://github.com/siliconflow/sd-team/assets/117806079/4c4a80a8-ccbf-4649-acee-1b7512b9bf13)   | ![ipadapter_advanced_oneflow](https://github.com/siliconflow/sd-team/assets/117806079/0f5e5ecf-4882-49ae-9792-00584aa1fcdd)  |

#### Performance Comparison

Timings for 30 steps at 1024*1024

| Accelerator           | Baseline (non-optimized) | OneDiff (optimized) | Percentage improvement |
| --------------------- | ------------------------ | ------------------- | ---------------------- |
| GeForce RTX 3090 | 7.27 s                   |  3.94 s      |     35.9%         |


### Ipadapter_weights
#### WorkFlow Description
source: https://github.com/cubiq/ComfyUI_IPAdapter_plus/blob/main/examples/ipadapter_weights.json

| Ipadapter_weights | Baseline (non-optimized)                                                                                         | OneDiff (optimized)                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| WorkFlow          |![Ipadapter_weights_torch](https://github.com/siliconflow/sd-team/assets/117806079/0eb8ae7f-cc1d-444b-9271-a5c0c464e93c)   | ![Ipadapter_weights_oneflow](https://github.com/siliconflow/sd-team/assets/117806079/4fbf13f0-d735-4064-aa29-807d45a20365) |

#### Performance Comparison

Timings for 30 steps at 1024*1024

| Accelerator           | Baseline (non-optimized) | OneDiff (optimized) | Percentage improvement |
| --------------------- | ------------------------ | ------------------- | ---------------------- |
| GeForce RTX 3090 |  41.20 s                   | 23.31  s              |  43.5 %                |

-----------------------------------------------------------------------

### ipadapter_style_composition
#### WorkFlow Description
source: https://github.com/cubiq/ComfyUI_IPAdapter_plus/blob/main/examples/ipadapter_style_composition.json

| ipadapter_style_composition_basic | Baseline (non-optimized)                                                                                         | OneDiff (optimized)                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| WorkFlow          |![ipadapter_style_composition_torch](https://github.com/siliconflow/sd-team/assets/117806079/2178d9ca-955e-4b99-b9a1-65f34e63906c)  | ![ipadapter_style_composition_oneflow](https://github.com/siliconflow/sd-team/assets/117806079/aba01f10-d809-4cae-987d-00dd99eabc08)  |


#### Performance Comparison

Timings for 30 steps at 1024*1024

| Accelerator           | Baseline (non-optimized) | OneDiff (optimized) | Percentage improvement |
| --------------------- | ------------------------ | ------------------- | ---------------------- |
| GeForce RTX 3090 | 8.72 s                   | 5.07  s           | 41.9%               |
  

<details close> 
<summary> Environment </summary>

- ComfyUI:
  - github: https://github.com/comfyanonymous/ComfyUI
  - commit: 2d4164271634476627aae31fbec251ca748a0ae0 
  - Date:   Wed May 15 02:40:06 2024 -0400
 
- ComfyUI_IPAdapter_plus:
  - github: https://github.com/cubiq/ComfyUI_IPAdapter_plus
  - commit 20125bf9394b1bc98ef3228277a31a3a52c72fc2 
  - Date:   Wed May 8 16:10:20 2024 +0200

- ComfyUI_InstantID:
  - github: https://github.com/cubiq/ComfyUI_InstantID
  - commit d8c70a0cd8ce0d4d62e78653674320c9c3084ec1 
  - Date:   Wed May 8 16:55:55 2024 +0200

- OneDiff:
  - github: https://github.com/siliconflow/onediff 

    ```shell
    # install onediff
    git clone https://github.com/siliconflow/onediff.git
    cd onediff && pip install -e .
    
    # install onediff_comfy_nodes
    ln -s $(pwd)/onediff_comfy_nodes path/to/ComfyUI/custom_nodes/
    # or
    # cp -r onediff_comfy_nodes path/to/ComfyUI/custom_nodes/
    ```
-  torch 2.3.0

</details> 


### Compatibility

| Functionality      | Supported |
| ------------------ | --------- |
| Dynamic Shape      | Yes       |
| Dynamic Batch Size | No        |

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.