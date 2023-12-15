# Bench OneDiff

## Build docker image

```bash
python3 -m pip install -r requirements.txt
python3 docker/main.py --yaml ./docker/config/community-default.yaml
```

## Prepare models

Download models from [here](#About-the-models). If you have downloaded the models before, please skip it.

## Run OneDiff Benchmark

To execute the OneDiff benchmarks (within the benchmark container), please follow these steps:

- Start docker container
  ```bash
  docker compose -f ./docker-compose.onediff:benchmark-community-default.yaml up -d
  ```

- Run benchmark
  ```bash
  cd /app/onediff/benchmarks && bash run_benchmark.sh /benchmark_model
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

You can obtain the models from [HuggingFace](https://huggingface.co) (excluding the int8 model) or download them from OSS (including the int8 model):

- Obtain and configure ossutil by executing the following commands:

  ```bash
  wget http://gosspublic.alicdn.com/ossutil/1.7.3/ossutil64  && chmod u+x ossutil64
  ```

- Configure ossutil by referring to [the official example](https://www.alibabacloud.com/help/en/oss/developer-reference/configure-ossutil?spm=a2c63.p38356.0.0.337f374a4pcwa4)
  ```bash
  ossutil64 config
  ```

- Download the benchmark models finally

  ```bash
  ./ossutil64 cp -r oss://oneflow-pro/onediff_benchmark_model/  benchmark_model  --update 
  ```
