# Bench OneDiff

## Build docker image

```bash
python3 -m pip install -r requirements.txt
python3 docker/main.py --yaml ./docker/config/community-default.yaml
```

## Prepare models

Download models from [here](# About the models). If you have downloaded the models before, please skip it.

## Run OneDiff Benchmark

To execute the OneDiff benchmarks (within the benchmark container), follow these steps:

```bash
cd /app/onediff/benchmarks/
bash run_benchmark.sh /benchmark_model
```

## About the models

The structure of `/benchmark_model` should follow this hierarchy:

```text
benchmark_model
├── stable-diffusion-2-1
├── stable-diffusion-v1-5
├── stable-diffusion-xl-base-1.0
├── stable-diffusion-xl-base-1.0-int8
```

You can obtain the models form [HuggingFace](https://huggingface.co) (excluding the int8 model) or download them from OSS (including the int8 model):

1. Obtain ossutil by executing the following command:

```bash
wget http://gosspublic.alicdn.com/ossutil/1.7.3/ossutil64  && chmod u+x ossutil64
```

2. Download the benchmark models:

```bash
./ossutil64 cp -r oss://oneflow-pro/onediff_benchmark_model/  benchmark_model  --update 
```
