#!/bin/bash

# Setup script for image generation model benchmarking

# For CN: Set environment variables for Hugging Face endpoint
export HF_ENDPOINT="https://hf-mirror.com"

REPO_ID="stabilityai/stable-diffusion-3-medium"

MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$REPO_ID'))")
if [ $? -ne 0 ]; then
    echo "Model download failed"
    exit 1
fi

echo "Model downloaded to: $MODEL_PATH"
