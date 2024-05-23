## nexfort backend for compiler in onediff
###  Dependency
```
pip3 install --pre -U torch==2.4.0.dev20240507 torchaudio==2.2.0.dev20240507+cu124 torchvision==0.19.0.dev20240507+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
pip3 install -U torchao==0.1
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
# model_id_or_path_to_PixArt-XL-2-1024-MS: /data/hf_models/PixArt-XL-2-1024-MS/ 
python3 ./benchmarks/text_to_image.py --model model_id_or_path_to_PixArt-XL-2-1024-MS --scheduler none --steps 20 --compiler nexfort --output-image ./pixart_alpha_nex.png
```
Performance on NVIDIA A100-PCIE-40GB:
- Warmup time: 771.418s
- Inference time: 2.045s
- Iterations per second: 10.743
- Max used CUDA memory: 13.855GiB
