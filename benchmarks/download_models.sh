#!/bin/bash
# set -x
set -e

die() { echo $1 && exit 1; }

download() {
  model=$1
  wget http://oneflow-pro.oss-cn-beijing.aliyuncs.com/models/${model}.zip && unzip -d .
  if [ $? != 0 ]; then
    die "failed to download model ${model}"
  fi
}

download stable-diffusion-v1-5
download stable-diffusion-2-1
download stable-diffusion-xl-base-1.0
download stable-diffusion-xl-base-1.0-int8
download stable-diffusion-xl-base-1.0-deepcache-int8
download stable-video-diffusion-img2vid-xt
download stable-video-diffusion-img2vid-xt-int8
download stable-video-diffusion-img2vid-xt-deepcache-int8

echo "all models are downloaded successfully!"
