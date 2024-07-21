#!/bin/bash
# set -x
set -e

MODEL_DIR=
WARMUPS=3
COMPILER=oneflow
OUTPUT_FILE=/dev/stdout

while getopts 'm:w:c:o:h' opt; do
  case "$opt" in
    m)
      MODEL_DIR=$OPTARG
      ;;

    w)
      WARMUPS=$OPTARG
      ;;

    c)
      COMPILER=$OPTARG
      ;;

    o)
      OUTPUT_FILE=$OPTARG
      ;;

    ?|h)
      echo "Usage: $(basename $0) [-m model_dir] [-w warmups] [-c compiler] [-o output_file]"
      echo "  -m model_dir: the directory of the models, if not set, use HF models"
      echo "  -w warmups: the number of warmups, default is `${WARMUPS}`"
      echo "  -c compiler: the compiler, default is `${COMPILER}`"
      echo "  -o output_file: the output file, default is `${OUTPUT_FILE}`"
      exit 1
      ;;
    *) echo "Unknown option ${opt}!" >&2
       exit 1 ;;
  esac
done

SCRIPT_DIR=$(realpath $(dirname $0))

SDXL_MODEL_PATH=stabilityai/stable-diffusion-xl-base-1.0
SDXL_LIGHT_PATH=ByteDance/SDXL-Lightning

if [ -z "${MODEL_DIR}" ]; then
  echo "model_dir unspecified, use HF models"
else
  echo "model_dir specified, use local models"
  MODEL_DIR=$(realpath ${MODEL_DIR})
  [ -d ${MODEL_DIR}/stable-diffusion-xl-base-1.0 ] && SDXL_MODEL_PATH=${MODEL_DIR}/stable-diffusion-xl-base-1.0
  [ -d ${MODEL_DIR}/SDXL-Lightning ] && SDXL_LIGHT_PATH=${MODEL_DIR}/SDXL-Lightning
fi

BENCHMARK_RESULT_TEXT="| Model | CPKT | Step | Is Lora | HxW | it/s | E2E Time (s) | CUDA Mem after (GiB) | Host Mem after (GiB) |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"


benchmark_sd_model_with_one_resolution() {
  model_name=$1
  model_path=$2
  repo_path=$3
  cpkt_path=$4
  warmups=$5
  compiler=$6
  height=$7
  width=$8
  echo "Run ${model_path} ${height}x${width}..."
  script_output=$(python3 ${SCRIPT_DIR}/text_to_image_sdxl_light.py --model ${model_path} --repo ${repo_path} --cpkt ${cpkt_path} --variant fp16 --warmups ${warmups} --compiler ${compiler} --height ${height} --width ${width} | tee /dev/tty)

  # Pattern to match:
  # Inference time: 0.560s
  # Iterations per second: 51.9
  # CUDA Mem after: 12.1GiB
  # Host Mem after: 5.3GiB

  inference_time=$(echo "${script_output}" | grep -oP '(?<=Inference time: )\d+\.\d+')
  iterations_per_second=$(echo "${script_output}" | grep -oP '(?<=Iterations per second: )\d+\.\d+')
  cuda_mem_after=$(echo "${script_output}" | grep -oP '(?<=CUDA Mem after: )\d+\.\d+')
  host_mem_after=$(echo "${script_output}" | grep -oP '(?<=Host Mem after: )\d+\.\d+')

  if [[ ${cpkt_path} =~ lora ]] ; then
    is_lora=True
  else
    is_lora=False
  fi
  steps=${cpkt_path#*sdxl_lightning_}
  steps=${steps%step*}

  BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}| ${model_name} | ${cpkt_path} | ${steps} | ${is_lora} | ${height}x${width} | ${iterations_per_second} | ${inference_time} | ${cuda_mem_after} | ${host_mem_after} |\n"
}

benchmark_sd_model() {
  model_name=$1
  model_path=$2
  repo_path=$3
  cpkt_path=$4
  resolutions=$5
  warmups=${WARMUPS}
  compiler=${COMPILER}
  for resolution in $(echo ${resolutions} | tr ',' ' '); do
    height=$(echo ${resolution} | cut -d'x' -f1)
    width=$(echo ${resolution} | cut -d'x' -f2)
    benchmark_sd_model_with_one_resolution ${model_name} ${model_path} ${repo_path} ${cpkt_path} ${warmups} ${compiler} ${height} ${width}
  done
}

benchmark_sd_model sdxl_light ${SDXL_MODEL_PATH} ${SDXL_LIGHT_PATH} sdxl_lightning_2step_unet.safetensors 1024x1024
benchmark_sd_model sdxl_light ${SDXL_MODEL_PATH} ${SDXL_LIGHT_PATH} sdxl_lightning_4step_unet.safetensors 1024x1024
benchmark_sd_model sdxl_light ${SDXL_MODEL_PATH} ${SDXL_LIGHT_PATH} sdxl_lightning_8step_unet.safetensors 1024x1024

benchmark_sd_model sdxl_light ${SDXL_MODEL_PATH} ${SDXL_LIGHT_PATH} sdxl_lightning_2step_lora.safetensors 1024x1024
benchmark_sd_model sdxl_light ${SDXL_MODEL_PATH} ${SDXL_LIGHT_PATH} sdxl_lightning_4step_lora.safetensors 1024x1024
benchmark_sd_model sdxl_light ${SDXL_MODEL_PATH} ${SDXL_LIGHT_PATH} sdxl_lightning_8step_lora.safetensors 1024x1024

echo -e "${BENCHMARK_RESULT_TEXT}" > ${OUTPUT_FILE}
