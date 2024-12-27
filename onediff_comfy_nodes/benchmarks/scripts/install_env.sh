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

# comfyui_controlnet_aux  ComfyUI_InstantID  ComfyUI_IPAdapter_plus  PuLID_ComfyUI
ln -s /share_nfs/hf_models/comfyui_resources/custom_nodes/* $CUSTOM_NODES/

echo "Installing dependencies..."
if [ "$CI" = "1" ]; then
  echo "Detected CI environment. Skipping local environment-specific dependencies."
else
  echo "Detected local environment. Installing local environment-specific dependencies."
  python3 -m pip install --user -r $CUSTOM_NODES/ComfyUI_InstantID/requirements.txt
  python3 -m pip install --user -r $CUSTOM_NODES/PuLID_ComfyUI/requirements.txt
fi

echo "Installing common dependencies..."
python3 -m pip install --user nexfort websocket-client==1.8.0 numpy==1.26.4 scikit-image
