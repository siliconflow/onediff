# OneDiff Benchmark

## Build docker image

```bash
python3 -m pip install -r requirements.txt
python3 docker/main.py --yaml ./docker/config/community-default.yaml
```

## Prepare models

```bash
sh download_models.sh models
```

## Run OneDiff Benchmark

```bash

docker run -it --rm --gpus all --shm-size 12g --ipc=host --security-opt seccomp=unconfined --privileged=true \
  -v `pwd`:/benchmark \
  onediff:benchmark-community-default \
  sh -c "cd /benchmark && sh run_all_benchmarks.sh -m models -o benchmark.md"
```

## Run Examples
### Run pixart alpha (with nexfort backend)
```
# model_id_or_path_to_PixArt-XL-2-1024-MS: /data/hf_models/PixArt-XL-2-1024-MS/ 
python3 text_to_image.py --model model_id_or_path_to_PixArt-XL-2-1024-MS --scheduler none --compiler nexfort
```
Performance on NVIDIA A100-PCIE-40GB:
Iterations per second of progress bar: 11.7
Inference time: 2.045s
Iterations per second: 10.517
CUDA Mem after: 13.569GiB

