#!/bin/bash

# This is a shell script to run the text-to-image benchmark.

# Activate virtual environment (if needed)
# source venv/bin/activate

# Run the Python script
# python3 scripts/text_to_image.py -w resources/example_workflow_api.json --comfy-pid 1438896
python3 scripts/text_to_image.py -w resources/sd3_workflow_api.json --comfy-pid 1463234
