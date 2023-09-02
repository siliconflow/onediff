
# Software Installation Guide 🚀 NEW

## Prerequisites

Before you begin, ensure that you meet the following prerequisites:

- NVIDIA driver with CUDA 11.7 support.
- Python environment with necessary packages, including `OneFlow`.

## Installation

### Clone the Repository and Install Requirements

1. Clone the repository and install the required packages from `requirements.txt` in a Python 3.10 environment, including `OneFlow` version 0.9.1 or higher:

    ```shell
    git clone https://github.com/Oneflow-Inc/diffusers.git
    cd diffusers && git checkout refactor-backend
    pip install .
    ```

### Install OneFlow

- (Optional) If you are located in China, you can speed up the download by setting up a pip mirror:

    ```bash
    python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    ```

- Install `OneFlow`:

    ```bash
    python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117
    ```

### Install PyTorch

You can install PyTorch using either Conda or Pip:

#### Using Conda (Recommended):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```


#### Using Pip:
```bash 
pip3 install torch torchvision torchaudio
```




