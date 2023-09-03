主要包括3块：
- [x] 1、安装和跑通例子的说明；
- [x] 2、新功能介绍，示例（SDXL） + 简要的原理；
- [ ] 3、性能指标（3090 + A100）


### Performance Comparison
Timings for 30 steps at 1024x1024
|                         | Baseline (torch.compile) | Optimized (oneflow.compile) | Percentage improvement |
| ----------------------- | ------------------------ | --------------------------- | ---------------------- |
| NVIDIA GeForce RTX 3090 | 7992 ms                  |                             |                        |
| A100 40G PCIE           | 4039 ms                  | 3295 ms                     | ~18%                   |
