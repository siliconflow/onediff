## nexfort backend for compiler in onediff
###  Dependency
```
pip3 install -U torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 torchao==0.1
```

### Install nexfort
```
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/sd/nexfort-0.1-cb3133ca2dae4265bc1d86068fc3aa1d.zip
unzip nexfort-0.1-cb3133ca2dae4265bc1d86068fc3aa1d.zip
cd nexfort-0.1-cb3133ca2dae4265bc1d86068fc3aa1d
pip3 install nexfort-0.1.dev195+torch230cu121-cp310-cp310-manylinux2014_x86_64.whl
```

### Run pixart alpha (with nexfort backend)

```
cd benchmarks
# model_id_or_path_to_PixArt-XL-2-1024-MS: /data/hf_models/PixArt-XL-2-1024-MS/ 
python3 text_to_image.py --model model_id_or_path_to_PixArt-XL-2-1024-MS --scheduler none --compiler nexfort
```
Performance on NVIDIA A100-PCIE-40GB:
Iterations per second of progress bar: 11.7
Inference time: 2.045s
Iterations per second: 10.517
CUDA Mem after: 13.569GiB
