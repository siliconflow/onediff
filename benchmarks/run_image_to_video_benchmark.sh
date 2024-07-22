#!/bin/bash
# set -x
set -e

MODEL_DIR=
WARMUPS=1
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

SVD_XT_MODEL_PATH=stabilityai/stable-video-diffusion-img2vid-xt

BENCHMARK_QUANT_MODEL=0
BENCHMARK_DEEP_CACHE_MODEL=0

if [ -z "${MODEL_DIR}" ]; then
  echo "model_dir unspecified, use HF models"
else
  echo "model_dir specified, use local models"
  MODEL_DIR=$(realpath ${MODEL_DIR})
  [ -d ${MODEL_DIR}/stable-video-diffusion-img2vid-xt ] && SVD_XT_MODEL_PATH=${MODEL_DIR}/stable-video-diffusion-img2vid-xt

  python3 -c "import onediff_quant" && echo "enable quant model" && BENCHMARK_QUANT_MODEL=1 || echo "disable quant model"
  # python3 -c "import onediffx" && echo "enable deepcache model" && BENCHMARK_DEEP_CACHE_MODEL=1 || echo "disable deepcache model"

  SVD_XT_QUANT_MODEL_PATH=${MODEL_DIR}/stable-video-diffusion-img2vid-xt-int8
  SVD_XT_DEEP_CACHE_QUANT_MODEL_PATH=${MODEL_DIR}/stable-video-diffusion-img2vid-xt-deepcache-int8
fi

BENCHMARK_RESULT_TEXT="| Model | HxW | it/s | E2E Time (s) | CUDA Mem after (GiB) | Host Mem after (GiB) |\n| --- | --- | --- | --- | --- | --- |\n"


benchmark_svd_model_with_one_resolution() {
  model_name=$1
  model_path=$2
  warmups=$3
  compiler=$4
  height=$5
  width=$6
  echo "Run ${model_path} ${height}x${width}..."
  if [[ ${model_name} =~ deepcache ]] ; then
    script_output=$(python3 ${SCRIPT_DIR}/image_to_video.py --model ${model_path} --variant fp16 --warmups ${warmups} --compiler ${compiler} --height ${height} --width ${width} --input-image ${SCRIPT_DIR}/resources/rocket.png --deepcache | tee /dev/tty)
  else
    script_output=$(python3 ${SCRIPT_DIR}/image_to_video.py --model ${model_path} --variant fp16 --warmups ${warmups} --compiler ${compiler} --height ${height} --width ${width} --input-image ${SCRIPT_DIR}/resources/rocket.png | tee /dev/tty)
  fi

  # Pattern to match:
  # Inference time: 0.560s
  # Iterations per second: 51.9
  # CUDA Mem after: 12.1GiB
  # Host Mem after: 5.3GiB

  inference_time=$(echo "${script_output}" | grep -oP '(?<=Inference time: )\d+\.\d+')
  iterations_per_second=$(echo "${script_output}" | grep -oP '(?<=Iterations per second: )\d+\.\d+')
  cuda_mem_after=$(echo "${script_output}" | grep -oP '(?<=CUDA Mem after: )\d+\.\d+')
  host_mem_after=$(echo "${script_output}" | grep -oP '(?<=Host Mem after: )\d+\.\d+')

  BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}| ${model_name} | ${height}x${width} | ${iterations_per_second} | ${inference_time} | ${cuda_mem_after} | ${host_mem_after} |\n"
}

benchmark_svd_model() {
  model_name=$1
  model_path=$2
  resolutions=$3
  warmups=${WARMUPS}
  compiler=${COMPILER}
  for resolution in $(echo ${resolutions} | tr ',' ' '); do
    height=$(echo ${resolution} | cut -d'x' -f1)
    width=$(echo ${resolution} | cut -d'x' -f2)
    benchmark_svd_model_with_one_resolution ${model_name} ${model_path} ${warmups} ${compiler} ${height} ${width}
  done
}

benchmark_svd_model svd_xt ${SVD_XT_MODEL_PATH} 576x1024

if [ ${BENCHMARK_QUANT_MODEL} != 0 ] && [ x"${COMPILER}" == x"oneflow" ]; then
  benchmark_svd_model svd_xt_quant ${SVD_XT_QUANT_MODEL_PATH} 576x1024
fi

if [ ${BENCHMARK_QUANT_MODEL} != 0 ] && [ ${BENCHMARK_DEEP_CACHE_MODEL} != 0 ] && [ x"${COMPILER}" == x"oneflow" ]; then
  benchmark_svd_model svd_xt_deepcache_quant ${SVD_XT_DEEP_CACHE_QUANT_MODEL_PATH} 576x1024
fi

echo -e "${BENCHMARK_RESULT_TEXT}" > ${OUTPUT_FILE}
