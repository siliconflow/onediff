#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  echo "Please provide the directory where ComfyUI should be installed."
  exit 1
fi

COMFYUI_ROOT=$1
CUSTOM_NODES=$COMFYUI_ROOT/custom_nodes

if [ ! -d "$COMFYUI_ROOT" ]; then
  echo "Error: Directory $COMFYUI_ROOT does not exist."
  exit 1
fi

ln -s /share_nfs/hf_models/comfyui_resources/custom_nodes/* $CUSTOM_NODES/

echo "Installing dependencies..."
pip install -r $COMFYUI_ROOT/requirements.txt --user
pip install websocket-client==1.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pynvml==11.5.0
# pip install numpy==1.26.4
pip install scikit-image

# pip install -r $CUSTOM_NODES/ComfyUI_InstantID/requirements.txt
# pip uninstall onnxruntime-gpu onnxruntime -y
# pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/