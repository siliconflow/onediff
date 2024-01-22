#!/bin/bash
# set -x
set -e

DOWNLOAD_PATH=./models

while getopts 'd:h' opt; do
  case ${opt} in
    d) DOWNLOAD_PATH=${OPTARG} ;;
    h) echo "Usage: $0 [-d download_path]" >&2
       exit 1 ;;
    *) echo "Unknown option ${opt}!" >&2
       exit 1 ;;
  esac
done

[ -z "${DOWNLOAD_PATH}" ] && echo "download_path unspecified" && exit 1

DOWNLOAD_PATH=$(realpath ${DOWNLOAD_PATH})
mkdir -p ${DOWNLOAD_PATH} || die "cannot create download directory: ${DOWNLOAD_PATH}."

die() { echo $1 && exit 1; }

download() {
  DOWNLOAD_PATH=$1
  model=$2
  if [ ! -d "${DOWNLOAD_PATH}/${model}" ]; then
    ( cd /tmp && wget -c http://oneflow-pro.oss-cn-beijing.aliyuncs.com/models/${model}.zip -O ${model}.zip && unzip ${model}.zip -d ${DOWNLOAD_PATH} && rm ${model}.zip )
    if [ $? != 0 ]; then
      die "failed to download model ${model}."
    fi
  else
    echo "model ${model} has been downloaded, will use the cached model."
  fi
}

download ${DOWNLOAD_PATH} stable-diffusion-v1-5
download ${DOWNLOAD_PATH} stable-diffusion-2-1
download ${DOWNLOAD_PATH} stable-diffusion-xl-base-1.0
download ${DOWNLOAD_PATH} stable-diffusion-xl-base-1.0-int8
download ${DOWNLOAD_PATH} stable-diffusion-xl-base-1.0-deepcache-int8
download ${DOWNLOAD_PATH} stable-video-diffusion-img2vid-xt
download ${DOWNLOAD_PATH} stable-video-diffusion-img2vid-xt-int8
download ${DOWNLOAD_PATH} stable-video-diffusion-img2vid-xt-deepcache-int8

echo "all models are downloaded successfully!"
