# OneDiff Benchmark

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
export BENCHMARK_MODEL_PATH=`pwd`/onediff_benchmark_model
docker compose -f ./docker-compose.onediff:benchmark-community-default.yaml up
```

Wait for a while, you will see the following logs,

```bash
onediff-benchmark-community-default  | Run SD1.5(FP16) 1024x1024...
onediff-benchmark-community-default  | + python3 ./text_to_image.py --model /benchmark_model/stable-diffusion-v1-5 --warmup 5 --height 1024 --width 1024
Loading pipeline components...:  29% 2/7 [00:00<00:00, 12.51it/s]`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["id2label"]` will be overriden.
Loading pipeline components...: 100% 7/7 [00:00<00:00, 12.91it/s]
100% 30/30 [00:43<00:00,  1.45s/it]  |
100% 30/30 [00:03<00:00,  8.47it/s]  |
100% 30/30 [00:03<00:00,  8.42it/s]  |
100% 30/30 [00:03<00:00,  8.41it/s]  |
100% 30/30 [00:03<00:00,  8.40it/s]  |
100% 30/30 [00:03<00:00,  8.38it/s]  |
onediff-benchmark-community-default  | e2e (30 steps) elapsed: 3.803581953048706 s, cuda memory usage: 7174.875 MiB
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

You can obtain the models from [HuggingFace](https://huggingface.co) (excluding the int8 model) or download them in a zip file from our server:

```bash
wget https://oneflow-pro.oss-cn-beijing.aliyuncs.com/onediff_benchmark_model.zip \
  -O onediff_benchmark_model.zip && \
  unzip ./onediff_benchmark_model.zip -d .
```

and **set the BENCHMARK_MODEL_PATH**:

```bash
export BENCHMARK_MODEL_PATH=`pwd`/onediff_benchmark_model
```


