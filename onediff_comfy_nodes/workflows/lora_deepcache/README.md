## LoRA to Combine with DeepCache for Acceleration

We combined OneDiff DeepCache and LoRA, achieving more than a 2.7x acceleration for shapes of 1024*1024, workflow reference:
![](lora-deepcache.png)

For the Aether Cloud style of LoRA, it can be downloaded from https://civitai.com/models/141029/aether-cloud-lora-for-sdxl.


Test results on A100:

|             | LoRA                | LoRA + DeepCache |
| ----------- | ------------------- | ------------------- |
| 512 * 1024  | 16.30 it/s (1x) | 39.57 it/s (2.43x) |
| 1024 * 1024 | 9.10 it/s (1x) | 25.19 it/s (2.77x) |

However, due to the characteristics of DeepCache, combining it with LoRA introduces some differences in the output:
![](compare.png)


Note: If your system or business requires more personalized support, please check: https://github.com/siliconflow/onediff?tab=readme-ov-file#onediff-enterprise-edition
