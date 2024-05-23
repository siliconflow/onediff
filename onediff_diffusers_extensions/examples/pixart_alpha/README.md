# Run PixArt alpha (with nexfort backend)
## Environment setup
### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

### Set up PixArt alpha
Using this model: https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS
Using this pipeline: https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart

## Run
model_id_or_path_to_PixArt-XL-2-1024-MS is the model id or model path of pixart alpha, such as `/data/hf_models/PixArt-XL-2-1024-MS/`

### Go to the onediff folder
```
cd onediff
```

### Run 1024*1024 without compile(the original pytorch HF diffusers pipelne)
```
python3 ./benchmarks/text_to_image.py --model /data/hf_models/PixArt-XL-2-1024-MS/ --scheduler none --steps 20 --compiler none --output-image ./pixart_alpha.png
```

### Run 1024*1024 with compile
```
python3 ./benchmarks/text_to_image.py --model /data/hf_models/PixArt-XL-2-1024-MS/ --scheduler none --steps 20 --compiler nexfort --output-image ./pixart_alpha.png
```

## Performance comparation
| Metric                               | NVIDIA A100-PCIE-40GB (1024 * 1024) |
| ------------------------------------ | ----------------------------------- |
| Data update date(yyyy-mm-dd)         | 2024-05-23                          |
| PyTorch iteration speed              | 8.469it/s                           |
| OneDiff iteration speed              | 10.51it/s(+24.2%)                   |
| PyTorch E2E time                     | 2.610s                              |
| OneDiff E2E time                     | 2.043s(-21.7%)                      |
| PyTorch Max Mem Used                 | 14.445GiB                           |
| OneDiff Max Mem Used                 | 13.571GiB                           |
| PyTorch Warmup with Run time         | 3.377s                              |
| OneDiff Warmup with Compilation time | 57.863s                             |
| OneDiff Warmup with Cache time       | TODO                                |

nexfort compile config: 
- {"mode": "max-autotune", "memory_format": "channels_last"}
- fuse_qkv_projections: True
