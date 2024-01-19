#!/bin/bash
# set -x
set -e

MODEL_DIR=
OUTPUT_FILE=/dev/stdout
WORK_DIR=/tmp

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

${SCRIPT_DIR}/download_models.sh -d ${MODEL_DIR}

BENCHMARK_RESULT_TEXT="# Benchmark report\n\n"

BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}## GPU Configuration\n\n"

BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\`\`\`\n$(nvidia-smi)\n\`\`\`\n\n"

BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}## Results\n\n"

TMP_OUTPUT_FILE=$(mktemp)

${SCRIPT_DIR}/run_text_to_image_benchmark.sh -m ${MODEL_DIR} -c none -o ${TMP_OUTPUT_FILE} && BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Text to Image (Vanilla PyTorch)\n\n$(cat ${TMP_OUTPUT_FILE})\n\n"
${SCRIPT_DIR}/run_text_to_image_benchmark.sh -m ${MODEL_DIR} -o ${TMP_OUTPUT_FILE} && BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Text to Image (OneFlow)\n\n$(cat ${TMP_OUTPUT_FILE})\n\n"
${SCRIPT_DIR}/run_text_to_image_benchmark_trt.sh -m ${MODEL_DIR} -o ${TMP_OUTPUT_FILE} -d ${WORK_DIR} && BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Text to Image (TensorRT)\n\n$(cat ${TMP_OUTPUT_FILE})\n\n"
${SCRIPT_DIR}/run_image_to_video_benchmark.sh -m ${MODEL_DIR} -c none -o ${TMP_OUTPUT_FILE} && BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Image to Video (Vanilla PyTorch)\n\n$(cat ${TMP_OUTPUT_FILE})\n\n"
${SCRIPT_DIR}/run_image_to_video_benchmark.sh -m ${MODEL_DIR} -o ${TMP_OUTPUT_FILE} && BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Image to Video (OneFlow)\n\n$(cat ${TMP_OUTPUT_FILE})\n\n"

echo -e "${BENCHMARK_RESULT_TEXT}" > ${OUTPUT_FILE}
