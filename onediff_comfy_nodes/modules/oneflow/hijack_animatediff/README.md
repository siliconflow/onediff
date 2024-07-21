## Accelerating ComfyUI-AnimateDiff-Evolved with OneDiff

The OneDiff framework in the community edition showcases an approximate 26.78% improvement in performance compared to PyTorch.

### Environment
Please Refer to the Readme in the Respective Repositories for Installation Instructions.

- ComfyUI:
  - github: https://github.com/comfyanonymous/ComfyUI
  - commit: `5d875d77fe6e31a4b0bc6dc36f0441eba3b6afe1`
  - Date:   `Wed Mar 20 20:48:54 2024 -0400`

- ComfyUI-AnimateDiff-Evolved:
  - github: https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
  - commit: `ab156a380fc5ac49ed92af5940f05942ce596296`
  - Date:   `Wed Mar 20 15:50:08 2024 -0500`

- OneDiff:
  - github: https://github.com/siliconflow/onediff


### Quick Start

> Recommend running the official example of ComfyUI AnimateDiff Evolved now, and then trying OneDiff acceleration.

Experiment (NVIDIA A100-PCIE-40GB) Workflow for OneDiff Acceleration in ComfyUI-AnimateDiff-Evolved:

1. Replace the **`Load Checkpoint`** node with **`Load Checkpoint - OneDiff`** node.
2. Add a **`Batch Size Patcher`** node before the **`Ksampler`** node (due to temporary lack of support for dynamic batch size).
As follows:

![animatediff](https://github.com/siliconflow/onediff/assets/109639975/f38251c8-1e75-4252-95eb-5fab0b45367d)

<details close>
<summary> Figure Notes </summary>

- NVIDIA: A100-PCIE-40GB
- Warmup: 2
  - PyTorch: **`56.52`** seconds
  - OneDiff: **`41.45`** seconds
  - percentage improvement：**~`26.78`** % 【`(56 - 41) / 56`】

- torch.__version__:`2.1.2+cu121`

</details>


Others workflows can be found in the following link:
- [Sample Workflows](
https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved?tab=readme-ov-file#samples-download-or-drag-images-of-the-workflows-into-comfyui-to-instantly-load-the-corresponding-workflows)

- [AnimateDiff-Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning#comfyui-usage)

### Compatibility

| Functionality      | Supported |
| ------------------ | --------- |
| Dynamic Shape      | Yes       |
| Dynamic Batch Size | No        |
| Vae SpeedUp        | No        |



[**`Community and Support`**](https://github.com/siliconflow/onediff?tab=readme-ov-file#community-and-support)
