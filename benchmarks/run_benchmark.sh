#!/bin/bash
set -e

# indicate which model to run
# e.g.  ./run_benchmark.sh sd15,sd21,sdxl or ./run_benchmark.sh all
run_model=$1



# set environment variables
export NEXFORT_GRAPH_CACHE=1
export NEXFORT_FX_FORCE_TRITON_SDPA=1


# model path
model_dir="/data1/hf_model"
sd15_path="${model_dir}/stable-diffusion-v1-5"
sd21_path="${model_dir}/stable-diffusion-2-1"
sdxl_path="${model_dir}/stable-diffusion-xl-base-1.0"
sd3_path="/data1/home/zhangxu/stable-diffusion-3-medium-diffusers"
flux_dev_path="${model_dir}/FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44"
flux_schnell_path="${model_dir}/FLUX.1-schnell"

# get current time
current_time=$(date +"%Y-%m-%d")
echo "Current time: ${current_time}"

# get NVIDIA GPU name
gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits | head -n 1 | sed 's/NVIDIA //; s/ /_/g')

# table header
BENCHMARK_RESULT_TEXT="| Data update date (yyyy-mm-dd) | GPU | Model | HxW | Compiler | Quantization | Iteration speed (it/s) | E2E Time (s) | Max used CUDA memory (GiB) | Warmup time (s) |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"


prompt="beautiful scenery nature glass bottle landscape, purple galaxy bottle"
quantize_config='{"quant_type": "fp8_e4m3_e4m3_dynamic_per_tensor"}'

# oneflow 没有compiler_config
#sd15_nexfort_compiler_config=""
#sd21_nexfort_compiler_config=""
#sdxl_nexfort_compiler_config=""

sd3_nexfort_compiler_config='{"mode": "max-optimize:max-autotune:low-precision:cache-all", "memory_format": "channels_last"}'
flux_nexfort_compiler_config='{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last"}'


# benchmark model with one resolution function
benchmark_model_with_one_resolution() {
  # model_name is the name of the model
  model_name=$1
  # model_path is the path of the model
  model_path=$2
  # steps is the number of inference steps
  steps=$3
  # compiler is the compiler used, e.g. none, oneflow, nexfort, transform
  compiler=$4
  # compiler_config is the compiler config used
  compiler_config=$5
  # height and width are the resolution of the image
  height=$6
  width=$7
  # quantize is whether to quantize
  quantize=$8

  echo "Running ${model_path} ${height}x${width}..."

  # if model_name contains sd3, use sd3 script
  if [[ "${model_name}" =~ sd3 ]]; then
    script_path="onediff_diffusers_extensions/examples/sd3/text_to_image_sd3.py"
  # if model_name contains flux, use flux script
  elif [[ "${model_name}" =~ flux ]]; then
    script_path="onediff_diffusers_extensions/examples/flux/text_to_image_flux.py"
  else
  # otherwise, use sd script
    script_path="benchmarks/text_to_image.py"
  fi

  # if quantize is True, add --quantize and --quantize-config
  if [[ ${quantize} == True ]]; then
    script_output=$(python3 ${script_path} \
      --model ${model_path} --variant fp16 --steps ${steps} \
      --height ${height} --width ${width} --seed 1 \
      --compiler ${compiler} --compiler-config "${compiler_config}" \
      --quantize --quantize-config "${quantize_config}" \
      --prompt "${prompt}" --print-output | tee /dev/tty)
  else
    script_output=$(python3 ${script_path} \
      --model ${model_path} --variant fp16 --steps ${steps} \
      --height ${height} --width ${width} --seed 1 \
      --compiler ${compiler} --compiler-config "${compiler_config}" \
      --prompt "${prompt}" --print-output | tee /dev/tty)
  fi

  # get inference time, iterations per second, max used cuda memory, warmup time
  inference_time=$(echo "${script_output}" | grep -oP '(?<=Inference time: )\d+\.\d+')
  iterations_per_second=$(echo "${script_output}" | grep -oP '(?<=Iterations per second: )\d+\.\d+')
  max_used_cuda_memory=$(echo "${script_output}" | grep -oP '(?<=Max used CUDA memory : )\d+\.\d+')
  warmup_time=$(echo "${script_output}" | grep -oP '(?<=Warmup time: )\d+\.\d+')

  # add benchmark result to BENCHMARK_RESULT_TEXT
  BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}| "${current_time}" | "${gpu_name}" | "${model_name}" | ${height}x${width} | ${compiler} | ${quantize} | ${iterations_per_second} | ${inference_time} | ${max_used_cuda_memory} | ${warmup_time} |\n"
}

# conda init
source ~/miniconda3/etc/profile.d/conda.sh

#########################################
# if run_model contains sd15 or all, run sd15
if [[ "${run_model}" =~ sd15|all ]]; then
  conda activate oneflow
  benchmark_model_with_one_resolution sd15 ${sd15_path} 30 none none 512 512 False
  benchmark_model_with_one_resolution sd15 ${sd15_path} 30 oneflow none 512 512 False
  benchmark_model_with_one_resolution sd15 ${sd15_path} 30 oneflow none 512 512 True
fi

# if run_model contains sd21 or all, run sd21
if [[ "${run_model}" =~ sd21|all ]]; then
  # activate oneflow environment
  conda activate oneflow
  benchmark_model_with_one_resolution sd21 ${sd21_path} 20 none none 768 768 False
  benchmark_model_with_one_resolution sd21 ${sd21_path} 20 oneflow none 768 768 False
  benchmark_model_with_one_resolution sd21 ${sd21_path} 20 oneflow none 768 768 True
fi

# if run_model contains sdxl or all, run sdxl
if [[ "${run_model}" =~ sdxl|all ]]; then
  # activate oneflow environment
  conda activate oneflow
  benchmark_model_with_one_resolution sdxl ${sdxl_path} 30 none none 1024 1024 False
  benchmark_model_with_one_resolution sdxl ${sdxl_path} 30 oneflow none 1024 1024 False
  benchmark_model_with_one_resolution sdxl ${sdxl_path} 30 oneflow none 1024 1024 True
fi
#########################################

#########################################
# if run_model contains sd3 or all, run sd3
if [[ "${run_model}" =~ sd3|all ]]; then
  conda activate nexfort
  # activate nexfort environment
  benchmark_model_with_one_resolution sd3 ${sd3_path} 28 none none 1024 1024 False
  benchmark_model_with_one_resolution sd3 ${sd3_path} 28 nexfort "${sd3_nexfort_compiler_config}" 1024 1024 False
  benchmark_model_with_one_resolution sd3 ${sd3_path} 28 nexfort "${sd3_nexfort_compiler_config}" 1024 1024 True
fi

# if run_model contains flux or all, run flux
if [[ "${run_model}" =~ flux|all ]]; then
  # activate nexfort environment
  conda activate nexfort
  benchmark_model_with_one_resolution flux_dev ${flux_dev_path} 20 none none 1024 1024 False
  benchmark_model_with_one_resolution flux_dev ${flux_dev_path} 20 nexfort "${flux_nexfort_compiler_config}" 1024 1024 False
  benchmark_model_with_one_resolution flux_dev ${flux_dev_path} 20 nexfort "${flux_nexfort_compiler_config}" 1024 1024 True
  benchmark_model_with_one_resolution flux_dev ${flux_dev_path} 20 transform none 1024 1024 False


  benchmark_model_with_one_resolution flux_schnell ${flux_schnell_path} 4 none none 1024 1024 False
  benchmark_model_with_one_resolution flux_schnell ${flux_schnell_path} 4 nexfort "${flux_nexfort_compiler_config}" 1024 1024 False
  benchmark_model_with_one_resolution flux_schnell ${flux_schnell_path} 4 nexfort "${flux_nexfort_compiler_config}" 1024 1024 True
  benchmark_model_with_one_resolution flux_schnell ${flux_schnell_path} 4 transform none 1024 1024 False
fi
#########################################


echo -e "\nBenchmark Results:"
# print benchmark result and add benchmark result to markdown file
echo -e ${BENCHMARK_RESULT_TEXT} | tee -a benchmark_result_"${gpu_name}".md
echo -e "\nBenchmark Done!"
