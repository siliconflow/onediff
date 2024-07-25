#!/bin/bash
# set -x
set -e

MODEL_DIR=./models
OUTPUT_FILE=/dev/stdout
WORK_DIR=./

while getopts 'm:o:d:h' opt; do
  case ${opt} in
    m) MODEL_DIR=${OPTARG} ;;
    o) OUTPUT_FILE=${OPTARG} ;;
    d) WORK_DIR=${OPTARG} ;;
    h) echo "Usage: $0 [-m model_dir] [-o output_file] [-d work_dir]" >&2
       exit 1 ;;
    *) echo "Unknown option ${opt}!" >&2
       exit 1 ;;
  esac
done

[ -z "${WORK_DIR}" ] && echo "work_dir unspecified" && exit 1
WORK_DIR=$(realpath ${WORK_DIR})

SCRIPT_DIR=$(realpath $(dirname $0))

BENCHMARK_RESULT_TEXT="# Benchmark report\n\n"

BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}## GPU Configuration\n\n"

BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\`\`\`\n$(nvidia-smi)\n\`\`\`\n\n"

BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}## Results\n\n"

OUTPUT_DIR=${WORK_DIR}/sdxl_light_output
mkdir -p ${OUTPUT_DIR}

TXT2IMG_TORCH_OUTPUT_FILE=${OUTPUT_DIR}/sdxl_light_torch.md
[ -f ${TXT2IMG_TORCH_OUTPUT_FILE} ] || ${SCRIPT_DIR}/run_text_to_image_sdxl_light_benchmark.sh -m ${MODEL_DIR} -c none -o ${TXT2IMG_TORCH_OUTPUT_FILE}
BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Text to Image (Vanilla PyTorch)\n\n$(cat ${TXT2IMG_TORCH_OUTPUT_FILE})\n\n"

TXT2IMG_ONEFLOW_OUTPUT_FILE=${OUTPUT_DIR}/sdxl_light_oneflow.md
[ -f ${TXT2IMG_ONEFLOW_OUTPUT_FILE} ] || ${SCRIPT_DIR}/run_text_to_image_sdxl_light_benchmark.sh -m ${MODEL_DIR} -o ${TXT2IMG_ONEFLOW_OUTPUT_FILE}
BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Text to Image (OneFlow)\n\n$(cat ${TXT2IMG_ONEFLOW_OUTPUT_FILE})\n\n"

echo -e "${BENCHMARK_RESULT_TEXT}" > ${OUTPUT_FILE}
