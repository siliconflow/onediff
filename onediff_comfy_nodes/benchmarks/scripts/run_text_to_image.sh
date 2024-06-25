#!/bin/bash

# This is a shell script to run the text-to-image benchmark.

# Activate virtual environment (if needed)
# source venv/bin/activate
export WORKFLOW_DIR=resources/workflows
# Run the Python script
python3 scripts/text_to_image.py -w $WORKFLOW_DIR/example_workflow_api.json --output-images
