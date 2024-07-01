# OneDiff Benchmark

In order to keep the code tidy and make it easier to use older versions, the old shell scripts have been placed in the directory `benchmarks/old_version`.  
You can follow the steps below to use the new version of OneDiff Benchmarks.  

1. [Prepare Docker Image](#prepare-docker-image)  
    - [Pull Docker Image from Aliyun](#pull-docker-image-from-aliyunrecommend)
    - [Build Docker Image from Scratch](#build-docker-image-from-scratch) 
2. [Prepare Models](#2-prepare-models)
3. [Start the Docker container](#3-start-the-docker-container)
4. [Run Benchmark Script](#4-run-benchmark-script)

## 1. Prepare Docker Image

### Pull Docker Image from Aliyun (Recommend)

After log in aliyun, pull docker image
```bash
docker pull registry.cn-beijing.aliyuncs.com/oneflow/benchmark-community-default:v1_864
```
### Build Docker Image from Scratch
```bash
python3 -m pip install -r requirements.txt
python3 docker/main.py --yaml ./docker/config/community-default.yaml
```

## 2. Prepare models

It is recommended to prepare the test model by creating a soft links between `/onediff/benchmarks/models` and model path `/share_nfs/hf_models/...`  

```bash
ln -s {original file path} {target file path}
ln -s /share_nfs/hf_models/stable-diffusion-v1-5 onediff/benchmarks/models
```
Or you can download models by running the following script.

```bash
sh download_models.sh models
```

## 3. Start the Docker Container

If you [pull image from aliyun](#pull-docker-image-from-aliyun-recommend), you can use the following script to start the docker container. 
```bash
docker run -it --rm \
    --gpus all \
    --shm-size=12g \
    --privileged=true \
    --ipc=host \
    -v $(pwd):/src/onediff \
    -w /src/onediff \
    registry.cn-beijing.aliyuncs.com/oneflow/benchmark-community-default:v1_864 \
    bash -c "python3 -m pip install -e . && cd onediff_diffusers_extensions && python3 -m pip install -e ."
```
The following ensures you are using the latest version of Onediff, including your changes.  
```bash
    bash -c "python3 -m pip install -e . && cd onediff_diffusers_extensions && python3 -m pip install -e ."
```

Or you just need change the image tag to start a container if you build image.  

```bash
docker run -it --rm \
    --gpus all \
    --shm-size=12g \
    --privileged=true \
    --ipc=host \
    -v $(pwd):/src/onediff \
    -w /src/onediff \
    onediff:benchmark-community-default \
    bash -c "python3 -m pip install -e . && cd onediff_diffusers_extensions && python3 -m pip install -e ."
```

## 4. Run Benchmark Script

### Run Test Script

```bash
cd benchmarks
python test_sd_benchmarks.py --model_dir models --model_name stable-diffusion-v1-5
```

use `python test_sd_benchmarks.py --help` to see more information.  

### Script Output

The output of test script `test_sd_benchmarks.py` includes the following parts:

```
oneflow version: 0.9.1.dev20240424+cu122
onediff version: 1.0.0.dev1
Inference time: 3.563s
Iterations per second: 8.812
CUDA Mem after: 16.880GiB
Host Mem after: 3.635GiB
```