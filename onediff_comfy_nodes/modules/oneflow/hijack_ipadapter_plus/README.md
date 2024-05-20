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

Experiment (NVIDIA A100-PCIE-40GB) Workflow for OneDiff Acceleration in ComfyUI_IPAdapter_plus:

1. Replace the **`Load Checkpoint`** node with **`Load Checkpoint - OneDiff`** node. 
2. Add a **`Batch Size Patcher`** node before the **`Ksampler`** node (due to temporary lack of support for dynamic batch size).
As follows:

![ipadapter_example](https://github.com/siliconflow/sd-team/assets/117806079/1940ec76-9247-43bc-b143-b646adc7c561)

![ipadapter_example int8](https://github.com/siliconflow/sd-team/assets/117806079/8633af92-cfc9-42b0-b265-55b85d5ffe2d)

 | PyTorch                                                                                                | OneDiff                                                                                                  |
 | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
 | ![torch](https://github.com/siliconflow/sd-team/assets/117806079/44872835-29e3-4df3-b178-49701400c1d6) | ![onediff](https://github.com/siliconflow/sd-team/assets/117806079/095c0e49-e554-4b23-92ec-0d0229b572f3) |

- NVIDIA A100-PCIE-40GB 
- batch_size 4
- warmup 4
- e2e
  - torch: 3.18 s (baseline)
  - onediff: 2.5 s (percentage improvement ~34.3%)
  - onediff+vae: 2.33 s (percentage improvement ~38.7%) 


### Compatibility

| Functionality      | Supported |
| ------------------ | --------- |
| Dynamic Shape      | Yes       |
| Dynamic Batch Size | No        |

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.