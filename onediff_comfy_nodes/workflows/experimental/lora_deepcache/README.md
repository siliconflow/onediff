## LoRA to Combine with DeepCache for Acceleration

We have designed the acceleration of workflows with LoRA using DeepCache, reference:
![](lora-deepcache.png)

For the Aether Cloud style of LoRA, it can be downloaded from https://civitai.com/models/141029/aether-cloud-lora-for-sdxl.


Test results on A100 GPU:

|             | LoRA                | LoRA with DeepCache |
| ----------- | ------------------- | ------------------- |
| 512 * 1024  | 1.43 s (16.30 it/s) | 0.70 s (39.57 it/s) |
| 1024 * 1024 | 2.63 s (9.10 it/s)  | 1.13 s (25.19 it/s) |


Note: If your system or business requires more personalized support, please send an email to contact@siliconflow.com.



