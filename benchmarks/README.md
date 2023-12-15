# Bench OneDiff

## Build docker image

```bash
python3 -m pip install -r requirements.txt
python3 docker/main.py --yaml ./docker/config/community-default.yaml
```

## Prepare models

Download models from [here](#About-the-models). If you have downloaded the models before, please skip it.

## Run OneDiff Benchmark

Start docker container and run the benchmark by the following command.

```bash
export BENCHMARK_MODEL_PATH=./benchmark_model
docker compose -f ./docker-compose.onediff:benchmark-community-default.yaml up
```

Wait for a while, you will see the following logs,

```bash
onediff-benchmark-community-default  | Run SD1.5(FP16) 1024x1024...
onediff-benchmark-community-default  | + python3 ./text_to_image.py --model /benchmark_model/stable-diffusion-v1-5 --warmup 5 --height 1024 --width 1024
Loading pipeline components...:  43% 3/7 [00:00<00:00, 20.94it/s]`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["id2label"]` will be overriden.
Loading pipeline components...: 100% 7/7 [00:00<00:00, 12.79it/s]
100% 30/30 [00:43<00:00,  1.45s/it]  |
100% 30/30 [00:03<00:00,  7.76it/s]  |
100% 30/30 [00:03<00:00,  7.74it/s]  |
100% 30/30 [00:03<00:00,  7.74it/s]  |
100% 30/30 [00:03<00:00,  7.72it/s]  |
100% 30/30 [00:03<00:00,  7.72it/s]  |
onediff-benchmark-community-default  | e2e (30 steps) elapsed: 4.1393163204193115 s, cuda memory usage: 7226.875 MiB
......
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

You can obtain the models from [HuggingFace](https://huggingface.co) (excluding the int8 model) or download them from OSS (including the int8 model). The OSS download method is as follows:

- Obtain ossutil by executing the following command:

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
