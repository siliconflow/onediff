## Accelerating ComfyUI_IPAdapter_plus with OneDiff
### Environment
Please Refer to the Readme in the Respective Repositories for Installation Instructions.

- ComfyUI:
  - github: https://github.com/comfyanonymous/ComfyUI
  - commit: 5d875d77fe6e31a4b0bc6dc36f0441eba3b6afe1 
  - Date:   Wed Mar 20 20:48:54 2024 -0400

- ComfyUI_IPAdapter_plus:
  - github: https://github.com/cubiq/ComfyUI_IPAdapter_plus
  - commit: 477217b364de427827f181cf11404bfc34181c41 
  - Date:   Tue Mar 26 15:02:27 2024 +0100
  
- OneDiff:
  - github: https://github.com/siliconflow/onediff 
  - branch: `git checkout dev_support_ipadapter_b`

### Quick Start

> Recommend running the official example of ComfyUI AnimateDiff Evolved now, and then trying OneDiff acceleration. 

Experiment (NVIDIA A100-PCIE-40GB) Workflow for OneDiff Acceleration in ComfyUI_IPAdapter_plus:

1. Replace the **`Load Checkpoint`** node with **`Load Checkpoint - OneDiff`** node. 
2. Add a **`Batch Size Patcher`** node before the **`Ksampler`** node (due to temporary lack of support for dynamic batch size).
As follows:

![workflow (1)](https://github.com/siliconflow/onediff/assets/109639975/b4c1bcac-1c15-45e0-94f1-7a352e2939fd)

 | PyTorch                                                                                                | OneDiff                                                                                                  |
 | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
 | ![torch](https://github.com/siliconflow/onediff/assets/109639975/b99838a6-2809-4e70-a4f2-966ba76c69d6) | ![onediff](https://github.com/siliconflow/onediff/assets/109639975/455741aa-d4e7-4b43-bfac-c5c52a66ac12) |

- NVIDIA A100-PCIE-40GB 
- batch_size 4
- warmup 4
- e2e
  - torch: 2.08 s
  - onediff: 1.35 s
  - percentage improvement：～35% 


### Compatibility

| Functionality      | Supported |
| ------------------ | --------- |
| Dynamic Shape      | Yes       |
| Dynamic Batch Size | No        |
| Vae Speed Up       | Yes       |

<div style="display: flex;">
<div style="flex: 1;">
        <img width="645" alt="image" src="https://github.com/siliconflow/onediff/assets/109639975/339e489e-aec7-488a-a242-276abfcf1cc3">
    </div>
    <div style="flex: 1;">
        <table>
            <thead>
                <tr>
                    <th>dynamic_modify</th>
                    <th></th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>weight</td>
                    <td>Yes</td>
                </tr>
                <tr>
                    <td>noise</td>
                    <td>Yes</td>
                </tr>
                <tr>
                    <td>weight_type</td>
                    <td>No</td>
                </tr>
                <tr>
                    <td>start_at</td>
                    <td>No</td>
                </tr>
                <tr>
                    <td>end_at</td>
                    <td>No</td>
                </tr>
                <tr> 
                    <td> unflod_batch </td>
                    <td> Untested </td>
            </tbody>
        </table>
  </div>
</div>

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.