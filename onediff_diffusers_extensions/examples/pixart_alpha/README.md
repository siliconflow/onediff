### Run pixart alpha (with nexfort backend)

```
# model_id_or_path_to_PixArt-XL-2-1024-MS: /data/hf_models/PixArt-XL-2-1024-MS/ 
# 1024*1024 without compile
python3 ./benchmarks/text_to_image.py --model /data/hf_models/PixArt-XL-2-1024-MS/ --scheduler none --steps 20 --compiler none --output-image ./pixart_alpha.png --height 1024 --width 1024

# 1024*1024 with compile
python3 ./benchmarks/text_to_image.py --model /data/hf_models/PixArt-XL-2-1024-MS/ --scheduler none --steps 20 --compiler nexfort --output-image ./pixart_alpha.png --height 1024 --width 1024

# 512*512 without compile
python3 ./benchmarks/text_to_image.py --model /data/hf_models/PixArt-XL-2-1024-MS/ --scheduler none --steps 20 --compiler none --output-image ./pixart_alpha.png --height 512 --width 512

# 512*512 with compile
python3 ./benchmarks/text_to_image.py --model /data/hf_models/PixArt-XL-2-1024-MS/ --scheduler none --steps 20 --compiler nexfort --output-image ./pixart_alpha.png --height 512 --width 512
```

| GPU                   | Image Resolution(30 steps) | PyTorch iteration speed | OneDiff iteration speed | PyTorch E2E time | OneDiff E2E time | PyTorch Mem | OneDiff Mem |
|-----------------------|----------------------------|-------------------------|-------------------------|------------------|------------------|-------------|-------------|
| NVIDIA A100-PCIE-40GB | 1024 * 1024                | 8.469                   | 10.51it/s               | 2.610s           | 2.043s           | 14.445GiB   | 13.571GiB   |
| NVIDIA A100-PCIE-40GB | 512 * 512                  | 8.480                   | 10.496it/s              | 2.593s           | 2.045s           | 14.445GiB   | 13.571GiB   |