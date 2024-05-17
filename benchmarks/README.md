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
