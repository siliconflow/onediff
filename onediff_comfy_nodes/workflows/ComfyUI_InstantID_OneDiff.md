## Quick Start

### Install

1. clone the repo

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ccssu/ComfyUI-InstantID.git 
```

2. install the dependencies

```bash
cd ComfyUI-InstantID
pip install -r requirements.txt
# install onediff and onediff_diffusers_extensions
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/community/cu121
git clone https://github.com/siliconflow/onediff.git
cd onediff
python3 -m pip install -e .
cd onediff_diffusers_extensions
python3 -m pip install -e .
```

3. Download relevant models [here](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID?tab=readme-ov-file#%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95--how-to-use)

## Accelerate

Add the 📷OneDiff Speed Up node as shown in the following diagram:：
![workflow (11)](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/109639975/f5bc8c9c-6bb2-4954-8d00-c6c512dddb53)

- warmup 2 times
- speed:
  - baseline: 2.77it/s
  - onediff: 5.18it/s
  - (5.18 - 2.77) / 2.77 = 0.87 (87% faster)
  
- e2e:
  - baseline: 20.18 s
  - onediff: 11.88 s
  -  (20.18 - 11.88) / 20.18 = 0.41 (41% faster)
 
## Environment

Tested on:(March 1st, 2023)
- NVIDIA A100-PCIE-40GB 
- Torch.__version__='2.1.2+cu121'
- Python = 3.10.13
- ComfyUI (Thu Feb 29 13:09:43 2024) commit: cb7c3a2921cfc0805be0229b4634e1143d60e6fe

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.