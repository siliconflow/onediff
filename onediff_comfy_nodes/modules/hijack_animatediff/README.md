## Accelerating ComfyUI-AnimateDiff-Evolved with OneDiff
### Environment
Please Refer to the Readme in the Respective Repositories for Installation Instructions.

- ComfyUI:
  - github: https://github.com/comfyanonymous/ComfyUI
  - commit: 38b7ac6e269e6ecc5bdd6fefdfb2fb1185b09c9d 

- ComfyUI-AnimateDiff-Evolved:
  - github: https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
  - commit: 4b31c0819d361e58f222cee3827079b3a6b6f966 

- OneDiff:
  - github: https://github.com/siliconflow/onediff 
  - branch: `git checkout dev_comfyui_animatediff_evolved`

### Quick Start

> Recommend running the official example of ComfyUI AnimateDiff Evolved now, and then trying OneDiff acceleration. 

Experiment (NVIDIA A100-PCIE-40GB) Workflow for OneDiff Acceleration in ComfyUI-AnimateDiff-Evolved:

1. Replace the **`Load Checkpoint`** node with **`Load Checkpoint - OneDiff`** node. 
2. Add a **`Batch Size Patcher`** node before the **`Ksampler`** node (due to temporary lack of support for dynamic batch size).
As follows:

![animatediff](https://github.com/siliconflow/onediff/assets/109639975/f38251c8-1e75-4252-95eb-5fab0b45367d)



Others workflows can be found in the following link:
- [Sample Workflows](
https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved?tab=readme-ov-file#samples-download-or-drag-images-of-the-workflows-into-comfyui-to-instantly-load-the-corresponding-workflows)

### Compatibility

| Functionality      | Supported |
| ------------------ | --------- |
| Dynamic Shape      | Yes       |
| Dynamic Batch Size | No        |
| Vae SpeedUp        | No        |


## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.