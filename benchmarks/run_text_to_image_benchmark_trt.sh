#!/bin/bash
# set -x
set -e

MODEL_DIR=
WARMUPS=3
OUTPUT_FILE=/dev/stdout
WORK_DIR=
TRT_VERSION=9.2
PROMPT="a beautiful photograph of Mt. Fuji during cherry blossom"

while getopts 'm:w:p:o:d:v:p:h' opt; do
  case "$opt" in
    m)
      MODEL_DIR=$OPTARG
      ;;

    w)
      WARMUPS=$OPTARG
      ;;

    p)
      PROMPT=$OPTARG
      ;;

    o)
      OUTPUT_FILE=$OPTARG
      ;;

    d)
      WORK_DIR=$OPTARG
      ;;

    v)
      TRT_VERSION=$OPTARG
      ;;

    ?|h)
      echo "Usage: $(basename $0) [-m model_dir] [-w warmups] [-p prompt] [-o output_file] [-d work_dir] [-v trt_version] [-h]"
      echo "  -m model_dir: the directory of the models, if not set, use HF models"
      echo "  -w warmups: the number of warmups, default is `${WARMUPS}`"
      echo "  -p prompt: the prompt, default is `${PROMPT}`"
      echo "  -o output_file: the output file, default is `${OUTPUT_FILE}`"
      echo "  -d work_dir: the work directory, default is `${WORK_DIR}`"
      echo "  -v trt_version: the version of TensorRT, default is `${TRT_VERSION}`"
      exit 1
      ;;
    *) echo "Unknown option ${opt}!" >&2
       exit 1 ;;
  esac
done

[ -z "${WORK_DIR}" ] && echo "work_dir unspecified" && exit 1
WORK_DIR=$(realpath ${WORK_DIR})

SCRIPT_DIR=$(realpath $(dirname $0))

SD15_MODEL_VERSION=1.5
SD21_MODEL_VERSION=2.1
SDXL_MODEL_VERSION=xl-1.0

TRT_VERSION_MAJOR=$(echo ${TRT_VERSION} | cut -d'.' -f1)
TRT_VERSION_MINOR=$(echo ${TRT_VERSION} | cut -d'.' -f2)
TRT_VERSION_NEXT=${TRT_VERSION_MAJOR}.$((${TRT_VERSION_MINOR}+1))

PYVENV_DIR=${WORK_DIR}/pyvenv
mkdir -p ${PYVENV_DIR}
TRT_PYVENV_DIR=${PYVENV_DIR}/trt_${TRT_VERSION}
if [ ! -d ${TRT_PYVENV_DIR} ]; then
  python3 -m venv ${TRT_PYVENV_DIR} --system-site-packages
fi
. ${TRT_PYVENV_DIR}/bin/activate

TRT_REPO_DIR=${WORK_DIR}/TensorRT
if [ ! -d ${TRT_REPO_DIR} ]; then
  git clone https://github.com/NVIDIA/TensorRT.git -b release/${TRT_VERSION} --single-branch ${TRT_REPO_DIR}
else
  cd ${TRT_REPO_DIR}
  if [ $(git branch --show-current) != release/${TRT_VERSION} ]; then
    git remote set-branches --add origin release/${TRT_VERSION}
    git checkout release/${TRT_VERSION}
    git pull
  fi
fi

python3 -m pip install --pre --extra-index-url https://pypi.nvidia.com "tensorrt>=${TRT_VERSION}.0,<${TRT_VERSION_NEXT}.0"

CUDA_VERSION=$(pip3 list | grep -oP '(?<=nvidia-cuda-runtime-cu)[0-9]+')
case ${CUDA_VERSION} in
  11)
    TORCH_CUDA_TAG=118
    ;;
  12)
    TORCH_CUDA_TAG=121
    ;;
  *)
    echo "Unsupported CUDA version: ${CUDA_VERSION}"
    exit 1
    ;;
esac

python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu${TORCH_CUDA_TAG}

cd $TRT_REPO_DIR/demo/Diffusion
python3 -m pip install -r requirements.txt

if [ ! -z "${MODEL_DIR}" ]; then
  echo "model_dir specified, use local models"
  MODEL_DIR=$(realpath ${MODEL_DIR})
  [ -d ${MODEL_DIR}/stable-diffusion-v1-5 ] && [ ! -d pytorch_model/${SD15_MODEL_VERSION}/TXT2IMG ] && mkdir -p pytorch_model/${SD15_MODEL_VERSION} && ln -s ${MODEL_DIR}/stable-diffusion-v1-5 pytorch_model/${SD15_MODEL_VERSION}/TXT2IMG
  [ -d ${MODEL_DIR}/stable-diffusion-2-1 ] && [ ! -d pytorch_model/${SD21_MODEL_VERSION}/TXT2IMG ] && mkdir -p pytorch_model/${SD21_MODEL_VERSION} && ln -s ${MODEL_DIR}/stable-diffusion-2-1 pytorch_model/${SD21_MODEL_VERSION}/TXT2IMG
  [ -d ${MODEL_DIR}/stable-diffusion-xl-base-1.0 ] && [ ! -d pytorch_model/${SDXL_MODEL_VERSION}/TXT2IMG ] && mkdir -p pytorch_model/${SDXL_MODEL_VERSION} && ln -s ${MODEL_DIR}/stable-diffusion-xl-base-1.0 pytorch_model/${SDXL_MODEL_VERSION}/TXT2IMG
fi

BENCHMARK_RESULT_TEXT="| Model | HxW | it/s | E2E Time (s) | CLIP Time (s) | UNet Time (s) | VAE-Dec Time (s) |\n| --- | --- | --- | --- | --- | --- | --- |\n"

benchmark_sd_model_with_one_resolution() {
  model_name=$1
  model_version=$2
  warmups=$3
  height=$4
  width=$5
  prompt="$6"
  onnx_dir="onnx_${model_version}"
  engine_dir="engine_${model_version}"
  if [[ ${model_name} =~ xl ]]; then
    script="demo_txt2img_xl.py"
  else
    script="demo_txt2img.py"
  fi
  echo "Run ${model_name} ${height}x${width}..."
  script_output=$(python3 ${script} "${prompt}" --build-dynamic-shape --onnx-dir ${onnx_dir} --engine-dir ${engine_dir} --version ${model_version} --num-warmup-runs ${warmups} --height ${height} --width ${width} | tee /dev/tty)

  # Pattern to match:
  # |-----------------|--------------|
  # |     Module      |   Latency    |
  # |-----------------|--------------|
  # |      CLIP       |      3.40 ms |
  # |    UNet x 30    |    495.92 ms |
  # |     VAE-Dec     |     24.88 ms |
  # |-----------------|--------------|
  # |    Pipeline     |    524.61 ms |
  # |-----------------|--------------|

  # grep: lookbehind assertion is not fixed length
  inference_time=$(echo "${script_output}" | grep 'Pipeline' | awk '{print $4}') && inference_time=$(python3 -c "print('{:.3f}'.format(${inference_time} / 1000))")
  clip_time=$(echo "${script_output}" | grep 'CLIP' | awk '{print $4}') && clip_time=$(python3 -c "print('{:.3f}'.format(${clip_time} / 1000))")
  unet_time=$(echo "${script_output}" | grep 'UNet' | awk '{print $6}') && unet_time=$(python3 -c "print('{:.3f}'.format(${unet_time} / 1000))")
  vae_dec_time=$(echo "${script_output}" | grep 'VAE-Dec' | awk '{print $4}') && vae_dec_time=$(python3 -c "print('{:.3f}'.format(${vae_dec_time} / 1000))")
  unet_steps=$(echo "${script_output}" | grep 'UNet' | awk '{print $4}')
  iterations_per_second=$(python3 -c "print('{:.3f}'.format(${unet_steps} / ${unet_time}))")
  BENCHMARK_RESULT_TEXT="${BENCHMARK_RESULT_TEXT}| ${model_name} | ${height}x${width} | ${iterations_per_second} | ${inference_time} | ${clip_time} | ${unet_time} | ${vae_dec_time} |\n"
}

benchmark_sd_model() {
  model_name=$1
  model_version=$2
  resolutions=$3
  warmups=${WARMUPS}
  prompt="${PROMPT}"
  for resolution in $(echo ${resolutions} | tr ',' ' '); do
    height=$(echo ${resolution} | cut -d'x' -f1)
    width=$(echo ${resolution} | cut -d'x' -f2)
    benchmark_sd_model_with_one_resolution ${model_name} ${model_version} ${warmups} ${height} ${width} "${prompt}"
  done
}

benchmark_sd_model sd15 ${SD15_MODEL_VERSION} 1024x1024,768x768,512x512
benchmark_sd_model sd21 ${SD21_MODEL_VERSION} 1024x1024,768x768,512x512
benchmark_sd_model sdxl ${SDXL_MODEL_VERSION} 1024x1024,768x768,512x512

echo -e "${BENCHMARK_RESULT_TEXT}" > ${OUTPUT_FILE}
