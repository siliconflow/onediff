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

OUTPUT_DIR=${WORK_DIR}/output
mkdir -p ${OUTPUT_DIR}

TXT2IMG_TORCH_OUTPUT_FILE=${OUTPUT_DIR}/txt2img_torch.md
[ -f ${TXT2IMG_TORCH_OUTPUT_FILE} ] || ${SCRIPT_DIR}/run_text_to_image_benchmark.sh -m ${MODEL_DIR} -c none -o ${TXT2IMG_TORCH_OUTPUT_FILE}
BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Text to Image (Vanilla PyTorch)\n\n$(cat ${TXT2IMG_TORCH_OUTPUT_FILE})\n\n"

TXT2IMG_ONEFLOW_OUTPUT_FILE=${OUTPUT_DIR}/txt2img_oneflow.md
[ -f ${TXT2IMG_ONEFLOW_OUTPUT_FILE} ] || ${SCRIPT_DIR}/run_text_to_image_benchmark.sh -m ${MODEL_DIR} -o ${TXT2IMG_ONEFLOW_OUTPUT_FILE}
BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Text to Image (OneFlow)\n\n$(cat ${TXT2IMG_ONEFLOW_OUTPUT_FILE})\n\n"

TXT2IMG_TRT_OUTPUT_FILE=${OUTPUT_DIR}/txt2img_trt.md
[ -f ${TXT2IMG_TRT_OUTPUT_FILE} ] || ${SCRIPT_DIR}/run_text_to_image_benchmark_trt.sh -o ${TXT2IMG_TRT_OUTPUT_FILE} -d ${WORK_DIR}
BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Text to Image (TensorRT)\n\n$(cat ${TXT2IMG_TRT_OUTPUT_FILE})\n\n"

IMG2VID_TORCH_OUTPUT_FILE=${OUTPUT_DIR}/img2vid_torch.md
[ -f ${IMG2VID_TORCH_OUTPUT_FILE} ] || ${SCRIPT_DIR}/run_image_to_video_benchmark.sh -m ${MODEL_DIR} -c none -o ${IMG2VID_TORCH_OUTPUT_FILE}
BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Image to Video (Vanilla PyTorch)\n\n$(cat ${IMG2VID_TORCH_OUTPUT_FILE})\n\n"

IMG2VID_ONEFLOW_OUTPUT_FILE=${OUTPUT_DIR}/img2vid_oneflow.md
[ -f ${IMG2VID_ONEFLOW_OUTPUT_FILE} ] || ${SCRIPT_DIR}/run_image_to_video_benchmark.sh -m ${MODEL_DIR} -o ${IMG2VID_ONEFLOW_OUTPUT_FILE}
BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}\n\n### Image to Video (OneFlow)\n\n$(cat ${IMG2VID_ONEFLOW_OUTPUT_FILE})\n\n"

echo -e "${BENCHMARK_RESULT_TEXT}" > ${OUTPUT_FILE}
