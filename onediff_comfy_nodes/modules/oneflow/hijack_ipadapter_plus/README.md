## Accelerating ComfyUI_IPAdapter_plus with OneDiff

### Quick Start

> Recommend running the official example of ComfyUI_IPAdapter_plus now, and then trying OneDiff acceleration.

Experiment (GeForce RTX 3090) Workflow for OneDiff Acceleration in ComfyUI_IPAdapter_plus:

1. Replace the **`Load Checkpoint`** node with **`Load Checkpoint - OneDiff`** node.
2. Add a **`Batch Size Patcher`** node before the **`Ksampler`** node (due to temporary lack of support for dynamic batch size).
As follows:
![workflow (19)](https://github.com/siliconflow/onediff/assets/117806079/07b153fd-a236-4c8d-a220-9b5823a79c17)


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


### Compatibility

| Functionality      | Supported |
| ------------------ | --------- |
| Dynamic Shape      | Yes       |
| Dynamic Batch Size | No        |

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.
