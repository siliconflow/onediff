# Run SDXL with nexfort backend (Beta Release)

1. [Environment Setup](#environment-setup)
   - [Set Up OneDiff](#set-up-onediff)
   - [Set Up NexFort Backend](#set-up-nexfort-backend)
   - [Set Up Diffusers Library](#set-up-diffusers)
   - [Set Up SDXL](#set-up-sdxl)
2. [Execution Instructions](#run)
   - [Run Without Compilation (Baseline)](#run-without-compilation-baseline)
   - [Run With Compilation](#run-with-compilation)
3. [Performance Comparison](#performance-comparison)
4. [Dynamic Shape for SDXL](#dynamic-shape-for-sdxl)
5. [Quality](#quality)

## Environment setup
### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

### Set up diffusers

```
pip3 install --upgrade diffusers[torch]
```
### Set up SDXL
Model version for diffusers: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

HF pipeline: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_xl.md

## Run

### Run without compilation (Baseline)
```shell
python3 benchmarks/text_to_image.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-xl.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler none \
  --variant fp16 \
  --seed 1 \
  --print-output
```

### Run with compilation

```shell
python3 benchmarks/text_to_image.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-xl-compile.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler nexfort \
  --compiler-config '{"mode": "benchmark:cudagraphs:max-autotune:low-precision:cache-all", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": true, "overrides.conv_benchmark": true, "overrides.matmul_allow_tf32": true}}' \
  --variant fp16 \
  --seed 1 \
  --print-output
```

## Performance comparison

Testing on NVIDIA GeForce RTX3090, RTX4090, A800-SXM4-80GB, with image size of 1024*1024, iterating 20 steps:
| Metric                               | RTX 3090  1024*1024   | RTX 4090 1024*1024    | A800-SXM4-80GB, 1024*1024 |
| ------------------------------------ | --------------------- | --------------------- | ------------------------- |
| Data update date (yyyy-mm-dd)        | 2024-07-31            | 2024-07-31            | 2024-07-31                |
| PyTorch iteration speed              | 3.97 it/s             | 8.44 it/s             | 9.92 it/s                 |
| OneDiff iteration speed              | 6.91 it/s (+75.2%)    | 14.77 it/s (+75.0%)   | 13.62 it/s (+37.3%)       |
| PyTorch E2E time                     | 5.72 s                | 2.76 s                | 2.28 s                    |
| OneDiff E2E time                     | 3.52 s (-38.5%)       | 1.56 s (-43.5%)       | 1.62 s (-28.9%)           |
| PyTorch Max Mem Used                 | 10.47 GiB             | 10.47 GiB             | 10.47 GiB                 |
| OneDiff Max Mem Used                 | 12.01 GiB             | 12.02 GiB             | 13.01 GiB                 |
| PyTorch Warmup with Run time         | 6.88 s                | 3.12 s                | 3.19 s                    |
| OneDiff Warmup with Compilation time | 511.35 s <sup>1</sup> | 250.91 s <sup>2</sup> | 308.81 s <sup>3</sup>     |
| OneDiff Warmup with Cache time       | 330.31 s              | 116.38 s              | 73.74 s                   |

<sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz. Note this is just for reference, and it varies a lot on different CPU.

<sup>2</sup> AMD EPYC 7543 32-Core Processor.

<sup>3</sup> Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz.

<details>
  <summary>Detailed Environment of RTX 3090</summary>

  ```
PyTorch version: 2.3.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OneFlow version: path: ['/home/wangyi/miniconda3/envs/py10/lib/python3.10/site-packages/oneflow'], version: 0.9.1.dev20240719+cu121, git_commit: dcaba7d, cmake_build_type: Release, rdma: True, mlir: True, enterprise: True
Nexfort version: 0.1.dev260
OneDiff version: 1.2.0.dev1+git.d3a5249e
OneDiffX version: 1.2.0.dev1+git.d3a5249e

OS: Ubuntu 20.04.5 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 10.0.0-4ubuntu1
CMake version: version 3.28.1
Libc version: glibc-2.31

Python version: 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.4.0-182-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.2.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA GeForce RTX 3090
GPU 1: NVIDIA GeForce RTX 3090
GPU 2: NVIDIA GeForce RTX 3090
GPU 3: NVIDIA GeForce RTX 3090
GPU 4: NVIDIA GeForce RTX 3090
GPU 5: NVIDIA GeForce RTX 3090
GPU 6: NVIDIA GeForce RTX 3090
GPU 7: NVIDIA GeForce RTX 3090

Nvidia driver version: 535.104.05
cuDNN version: Probably one of the following:
/usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn.so.8
/usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
/usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
/usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
/usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
/usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
/usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                         Little Endian
Address sizes:                      46 bits physical, 57 bits virtual
CPU(s):                             64
On-line CPU(s) list:                0-63
Thread(s) per core:                 2
Core(s) per socket:                 16
Socket(s):                          2
NUMA node(s):                       2
Vendor ID:                          GenuineIntel
CPU family:                         6
Model:                              106
Model name:                         Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz
Stepping:                           6
Frequency boost:                    enabled
CPU MHz:                            918.199
CPU max MHz:                        3400.0000
CPU min MHz:                        800.0000
BogoMIPS:                           4800.00
Virtualization:                     VT-x
L1d cache:                          1.5 MiB
L1i cache:                          1 MiB
L2 cache:                           40 MiB
L3 cache:                           48 MiB
NUMA node0 CPU(s):                  0-15,32-47
NUMA node1 CPU(s):                  16-31,48-63
Vulnerability Gather data sampling: Mitigation; Microcode
Vulnerability Itlb multihit:        Not affected
Vulnerability L1tf:                 Not affected
Vulnerability Mds:                  Not affected
Vulnerability Meltdown:             Not affected
Vulnerability Mmio stale data:      Mitigation; Clear CPU buffers; SMT vulnerable
Vulnerability Retbleed:             Not affected
Vulnerability Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:           Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:           Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
Vulnerability Srbds:                Not affected
Vulnerability Tsx async abort:      Not affected
Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 invpcid_single ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local wbnoinvd dtherm ida arat pln pts avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq rdpid md_clear pconfig flush_l1d arch_capabilities

Versions of relevant libraries:
[pip3] diffusers==0.29.2
[pip3] diffusers-extensions==0.1.0
[pip3] flake8==7.0.0
[pip3] mypy==1.10.0
[pip3] mypy-extensions==1.0.0
[pip3] numpy==1.26.2
[pip3] open-clip-torch==2.20.0
[pip3] pytorch-lightning==1.9.4
[pip3] torch==2.3.1
[pip3] torchao==0.1
[pip3] torchaudio==2.3.1
[pip3] torchdiffeq==0.2.3
[pip3] torchmetrics==1.2.1
[pip3] torchsde==0.2.6
[pip3] torchvision==0.18.1
[pip3] transformers==4.42.4
[pip3] triton==2.3.1
[conda] Could not collect
  ```
</details>


<details>
  <summary>Detailed Environment of RTX 4090</summary>

  ```
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OneFlow version: path: ['/home/wangyi/miniconda3/envs/py10/lib/python3.10/site-packages/oneflow'], version: 0.9.1.dev20240727+cu122, git_commit: f230775, cmake_build_type: Release, rdma: True, mlir: True, enterprise: True
Nexfort version: 0.1.dev260
OneDiff version: 1.1.1.dev65+gf50c02b4
OneDiffX version: 1.1.1.dev65+gf50c02b4

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-92-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090
GPU 2: NVIDIA GeForce RTX 4090
GPU 3: NVIDIA GeForce RTX 4090
GPU 4: NVIDIA GeForce RTX 4090
GPU 5: NVIDIA GeForce RTX 4090
GPU 6: NVIDIA GeForce RTX 4090
GPU 7: NVIDIA GeForce RTX 4090

Nvidia driver version: 550.90.07
cuDNN version: Probably one of the following:
/usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudnn.so.8
/usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
/usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
/usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
/usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
/usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
/usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      48 bits physical, 48 bits virtual
Byte Order:                         Little Endian
CPU(s):                             128
On-line CPU(s) list:                0-127
Vendor ID:                          AuthenticAMD
Model name:                         AMD EPYC 7543 32-Core Processor
CPU family:                         25
Model:                              1
Thread(s) per core:                 2
Core(s) per socket:                 32
Socket(s):                          2
Stepping:                           1
Frequency boost:                    enabled
CPU max MHz:                        3737.8899
CPU min MHz:                        1500.0000
BogoMIPS:                           5590.02
Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload vgif v_spec_ctrl umip pku ospke vaes vpclmulqdq rdpid overflow_recov succor smca fsrm
Virtualization:                     AMD-V
L1d cache:                          2 MiB (64 instances)
L1i cache:                          2 MiB (64 instances)
L2 cache:                           32 MiB (64 instances)
L3 cache:                           512 MiB (16 instances)
NUMA node(s):                       8
NUMA node0 CPU(s):                  0-7,64-71
NUMA node1 CPU(s):                  8-15,72-79
NUMA node2 CPU(s):                  16-23,80-87
NUMA node3 CPU(s):                  24-31,88-95
NUMA node4 CPU(s):                  32-39,96-103
NUMA node5 CPU(s):                  40-47,104-111
NUMA node6 CPU(s):                  48-55,112-119
NUMA node7 CPU(s):                  56-63,120-127
Vulnerability Gather data sampling: Not affected
Vulnerability Itlb multihit:        Not affected
Vulnerability L1tf:                 Not affected
Vulnerability Mds:                  Not affected
Vulnerability Meltdown:             Not affected
Vulnerability Mmio stale data:      Not affected
Vulnerability Retbleed:             Not affected
Vulnerability Spec rstack overflow: Mitigation; safe RET
Vulnerability Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:           Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:           Mitigation; Retpolines, IBPB conditional, IBRS_FW, STIBP always-on, RSB filling, PBRSB-eIBRS Not affected
Vulnerability Srbds:                Not affected
Vulnerability Tsx async abort:      Not affected

Versions of relevant libraries:
[pip3] diffusers==0.29.2
[pip3] numpy==1.26.2
[pip3] open-clip-torch==2.20.0
[pip3] pytorch-lightning==1.9.4
[pip3] torch==2.3.0
[pip3] torchdiffeq==0.2.3
[pip3] torchmetrics==1.3.2
[pip3] torchsde==0.2.6
[pip3] torchvision==0.18.1
[pip3] transformers==4.30.2
[pip3] triton==2.3.0
[conda] numpy                     1.26.2                   pypi_0    pypi
[conda] open-clip-torch           2.20.0                   pypi_0    pypi
[conda] pytorch-lightning         1.9.4                    pypi_0    pypi
[conda] torch                     2.3.0                    pypi_0    pypi
[conda] torchdiffeq               0.2.3                    pypi_0    pypi
[conda] torchmetrics              1.3.2                    pypi_0    pypi
[conda] torchsde                  0.2.6                    pypi_0    pypi
[conda] torchvision               0.18.1                   pypi_0    pypi
[conda] triton                    2.3.0                    pypi_0    pypi
  ```
</details>

<details>
  <summary>Detailed Environment of A800-SXM4-80GB</summary>

  ```
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OneFlow version: path: ['/home/wangyi/miniconda3/envs/py10/lib/python3.10/site-packages/oneflow'], version: 0.9.1.dev20240724+cu122, git_commit: dcaba7d, cmake_build_type: Release, rdma: True, mlir: True, enterprise: True
Nexfort version: 0.1.dev260
OneDiff version: 1.2.1.dev7+gfc3de4e9
OneDiffX version: 1.2.1.dev7+gfc3de4e9

OS: Ubuntu 22.04.2 LTS (x86_64)
GCC version: (Ubuntu 11.3.0-1ubuntu1~22.04.1) 11.3.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-60-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA A800-SXM4-80GB
GPU 1: NVIDIA A800-SXM4-80GB
GPU 2: NVIDIA A800-SXM4-80GB
GPU 3: NVIDIA A800-SXM4-80GB
GPU 4: NVIDIA A800-SXM4-80GB
GPU 5: NVIDIA A800-SXM4-80GB
GPU 6: NVIDIA A800-SXM4-80GB
GPU 7: NVIDIA A800-SXM4-80GB

Nvidia driver version: 535.54.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Address sizes:                   46 bits physical, 57 bits virtual
Byte Order:                      Little Endian
CPU(s):                          128
On-line CPU(s) list:             0-127
Vendor ID:                       GenuineIntel
Model name:                      Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz
CPU family:                      6
Model:                           106
Thread(s) per core:              2
Core(s) per socket:              32
Socket(s):                       2
Stepping:                        6
CPU max MHz:                     3400.0000
CPU min MHz:                     800.0000
BogoMIPS:                        5200.00
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 invpcid_single intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid fsrm md_clear pconfig flush_l1d arch_capabilities
Virtualization:                  VT-x
L1d cache:                       3 MiB (64 instances)
L1i cache:                       2 MiB (64 instances)
L2 cache:                        80 MiB (64 instances)
L3 cache:                        96 MiB (2 instances)
NUMA node(s):                    2
NUMA node0 CPU(s):               0-31,64-95
NUMA node1 CPU(s):               32-63,96-127
Vulnerability Itlb multihit:     Not affected
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Mmio stale data:   Mitigation; Clear CPU buffers; SMT vulnerable
Vulnerability Retbleed:          Not affected
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Not affected

Versions of relevant libraries:
[pip3] diffusers==0.26.0
[pip3] numpy==1.26.4
[pip3] torch==2.3.0
[pip3] transformers==4.42.0
[pip3] triton==2.3.0
[conda] numpy                     1.26.4                   pypi_0    pypi
[conda] torch                     2.3.0                    pypi_0    pypi
[conda] triton                    2.3.0                    pypi_0    pypi

  ```
</details>

## Dynamic shape for SDXL

Add `"dynamic": true` in --compiler-config.

Run:

```shell
python3 benchmarks/text_to_image.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --height 1024 --width 1024 \
  --scheduler none \
  --steps 20 \
  --output-image ./stable-diffusion-xl-compile.png \
  --prompt "beautiful scenery nature glass bottle landscape, , purple galaxy bottle," \
  --compiler nexfort \
  --compiler-config '{"mode": "cudagraphs:max-autotune:low-precision:cache-all", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": true, "overrides.conv_benchmark": true, "overrides.matmul_allow_tf32": true}, "dynamic": true}' \
  --run_multiple_resolutions 1
```

## Quality
When using nexfort as the backend for onediff compilation acceleration, the generated images are lossless.

<p align="center">
<img src="../../../imgs/nexfort_sdxl_demo.png">
</p>

## Note
If you encounter an error like below, please set the `"inductor.optimize_linear_epilogue": true` to `"inductor.optimize_linear_epilogue": false` in --compiler-config.

```
torch._dynamo.exc.BackendCompilerFailed: backend='nexfort' raised:
ErrorFromChoice: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasLtMatmul( ltHandle, computeDesc.descriptor(), &alpha_val, mat1_ptr, Adesc.descriptor(), mat2_ptr, Bdesc.descriptor(), &beta_val, mat3_ptr, Ddesc.descriptor(), result_ptr, Cdesc.descriptor(), &heuristicResult.algo, workspace.data_ptr(), workspaceSize, at::cuda::getCurrentCUDAStream())`
From choice ExternKernelCaller(extern_kernels.nexfort_cuda_linear_epilogue)
inputs = [
    torch.empty_strided((2, 10, 4096, 64), (2621440, 64, 640, 1), dtype=torch.float16, device='cuda'),
    torch.empty_strided((640, 640), (640, 1), dtype=torch.float16, device='cuda'),
    torch.empty_strided((640,), (1,), dtype=torch.float16, device='cuda'),
    torch.empty_strided((2, 4096, 640), (2621440, 640, 1), dtype=torch.float16, device='cuda'),
]
out = torch.empty_strided((2, 4096, 640), (2621440, 640, 1), dtype=torch.float16, device='cuda')
```
