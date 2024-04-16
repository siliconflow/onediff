## Accelerating ComfyUI_IPAdapter_plus with OneDiff
### Environment
Please Refer to the Readme in the Respective Repositories for Installation Instructions.

- ComfyUI:
  - github: https://github.com/comfyanonymous/ComfyUI
  - commit: 4bd7d55b9028d79829a645edfe8259f7b7a049c0 
  - Date: Thu Apr 11 22:43:05 2024 -0400
  
- ComfyUI_IPAdapter_plus:
  - github: https://github.com/cubiq/ComfyUI_IPAdapter_plus
  - commit 417d806e7a2153c98613e86407c1941b2b348e88 
  - Date:  Wed Apr 10 13:28:41 2024 +0200
  
- OneDiff:
  - github: https://github.com/siliconflow/onediff 

### Quick Start

> Recommend running the official example of ComfyUI_IPAdapter_plus now, and then trying OneDiff acceleration. 

Experiment (NVIDIA A100-PCIE-40GB) Workflow for OneDiff Acceleration in ComfyUI_IPAdapter_plus:

1. Replace the **`Load Checkpoint`** node with **`Load Checkpoint - OneDiff`** node. 
2. Add a **`Batch Size Patcher`** node before the **`Ksampler`** node (due to temporary lack of support for dynamic batch size).
As follows:

![ipadapter_example](https://github.com/siliconflow/onediff/assets/109639975/adb2df92-b0f0-4650-ae02-5fd458209b92)



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

<!-- <div style="display: flex;">
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
                    <td>No</td>
                </tr>
                <tr>
                    <td>noise</td>
                    <td>No</td>
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
</div> -->

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.