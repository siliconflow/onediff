#!/bin/bash

# This is a shell script to run the text-to-image benchmark.

# Activate virtual environment (if needed)
# source venv/bin/activate

# Run the Python script
# python3 scripts/text_to_image.py -w resources/example_workflow_api.json
# python3 scripts/text_to_image.py -w  resources/baseline/sd3_baseline.json  --output-images
# python3 scripts/text_to_image.py -w  resources/nexfort/sd3_unet_vae_speedup.json --output-images
python3 scripts/text_to_image.py \
    -w resources/baseline/lora.json resources/baseline/lora_multiple.json \
    --output-images 
    # --baseline-dir /home/fengwen/worksplace/packages/onediff/onediff_comfy_nodes/benchmarks/results/exp12/imgs
