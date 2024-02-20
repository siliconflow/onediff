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

### Quick Start

> Recommend running the official example of ComfyUI AnimateDiff Evolved now, and then trying OneDiff acceleration. 

Experiment Workflow for OneDiff Acceleration in ComfyUI-AnimateDiff-Evolved , Replace the "Load Checkpoint" node with "Load Checkpoint - OneDiff" node. 
As follows:
![workflow](https://github.com/siliconflow/onediff/assets/109639975/e94877c1-eb1e-464a-9b9e-731bef02aca3)

Others workflows can be found in the following link:
- [Sample Workflows](
https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved?tab=readme-ov-file#samples-download-or-drag-images-of-the-workflows-into-comfyui-to-instantly-load-the-corresponding-workflows)

### Compatibility
| Functionality      | Supported |
| ------------------ | --------- |
| Dynamic Shape      | Yes       |
| Dynamic Batch Size | No        |

