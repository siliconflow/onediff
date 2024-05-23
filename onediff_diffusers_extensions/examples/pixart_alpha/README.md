# Run PixArt alpha (with nexfort backend)
## Environment setup
### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

### Set up PixArt alpha
HF model: https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS

HF pipeline: https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart

## Run
model_id_or_path_to_PixArt-XL-2-1024-MS is the model id or model path of pixart alpha, such as `/data/hf_models/PixArt-XL-2-1024-MS/`

### Go to the onediff folder
```
cd onediff
```

### Run 1024*1024 without compile(the original pytorch HF diffusers pipeline)
```
python3 ./benchmarks/text_to_image.py --model /data/hf_models/PixArt-XL-2-1024-MS/ --scheduler none --steps 20 --compiler none --output-image ./pixart_alpha.png
```

### Run 1024*1024 with compile
```
python3 ./benchmarks/text_to_image.py --model /data/hf_models/PixArt-XL-2-1024-MS/ --scheduler none --steps 20 --compiler nexfort --output-image ./pixart_alpha.png
```

## Performance comparation
### nexfort compile config
- compiler-config default is `{"mode": "max-optimize:max-autotune:freezing:benchmark:cudagraphs", "memory_format": "channels_last"}` in `/benchmarks/text_to_image.py`
  - setting `--compiler-config '{"mode": "max-autotune", "memory_format": "channels_last"}'` will reduce compilation time to 57.863s and just slightly reduce the performance
- fuse_qkv_projections: True

### Metric
| Metric                               | NVIDIA A100-PCIE-40GB (1024 * 1024) |
| ------------------------------------ | ----------------------------------- |
| Data update date(yyyy-mm-dd)         | 2024-05-23                          |
| PyTorch iteration speed              | 8.623it/s                           |
| OneDiff iteration speed              | 10.743it/s(+24.58%)                 |
| PyTorch E2E time                     | 2.568s                              |
| OneDiff E2E time                     | 1.992s(-22.4%)                      |
| PyTorch Max Mem Used                 | 14.445GiB                           |
| OneDiff Max Mem Used                 | 13.855GiB                           |
| PyTorch Warmup with Run time         | 4.100s                              |
| OneDiff Warmup with Compilation time | 771.418s                            |
| OneDiff Warmup with Cache time       | TODO                                |

Note:
- OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz. Note this is just for reference, and it varies a lot on different CPU.

