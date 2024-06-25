#!/bin/bash
set -e

echo "Starting test execution..."

echo "Running run_oneflow_case_ci.sh"
bash scripts/run_oneflow_case_ci.sh

echo "Running run_nexfort_case_ci.sh"
bash scripts/run_nexfort_case_ci.sh

if [ "$CI" = "1" ]; then
    echo "Detected CI environment. Skipping run_oneflow_case_local.sh."
else
    echo "Running run_oneflow_case_local.sh"
    bash scripts/run_oneflow_case_local.sh
fi

echo "All tests have been executed successfully."
