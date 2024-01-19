#!/bin/bash
# set -x
set -e

download_path=./models

die() { echo $1 && exit 1; }

download() {
  download_path=$1
  model=$2
  if [ ! -d "${download_path}/${model}" ]; then
    wget http://oneflow-pro.oss-cn-beijing.aliyuncs.com/models/${model}.zip && unzip ${model}.zip -d ${download_path}
    if [ $? != 0 ]; then
      die "failed to download model ${model}."
    fi
  else
    echo "model ${model} has been downloaded, will use the cached model."
  fi
}

if [ "$#" -ge 1 ]; then
  download_path=$1
fi

if [ ! -d "${download_path}" ]; then
  mkdir -p ${download_path} || die "cannot create download directory: ${download_path}."
fi

download ${download_path} stable-diffusion-v1-5
download ${download_path} stable-diffusion-2-1
download ${download_path} stable-diffusion-xl-base-1.0
download ${download_path} stable-diffusion-xl-base-1.0-int8
download ${download_path} stable-diffusion-xl-base-1.0-deepcache-int8
download ${download_path} stable-video-diffusion-img2vid-xt
download ${download_path} stable-video-diffusion-img2vid-xt-int8
download ${download_path} stable-video-diffusion-img2vid-xt-deepcache-int8

echo "all models are downloaded successfully!"
