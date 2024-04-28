#!/usr/bin/env python3

import argparse
import os
import re
import subprocess

def benchmark_sd_model_with_one_resolution(model_name, model_path, repo_path, cpkt_path, warmups, compiler, height, width):
  print(f"Run {model_path} {height}x{width}...")
  script_output = subprocess.check_output([
    "python3",
    os.path.join(SCRIPT_DIR, "text_to_image_sdxl_light.py"),
    "--model", model_path,
    "--repo", repo_path,
    "--cpkt", cpkt_path,
    "--variant", "fp16",
    "--warmups", str(warmups),
    "--compiler", compiler,
    "--height", str(height),
    "--width", str(width)
  ]).decode("utf-8")

  # Pattern to match:
  # Inference time: 0.560s
  # Iterations per second: 51.9
  # CUDA Mem after: 12.1GiB
  # Host Mem after: 5.3GiB

  inference_time = re.search(r'Inference time: (\d+\.\d+)', script_output).group(1)
  iterations_per_second = re.search(r'Iterations per second: (\d+\.\d+)', script_output).group(1)
  cuda_mem_after = re.search(r'CUDA Mem after: (\d+\.\d+)', script_output).group(1)
  host_mem_after = re.search(r'Host Mem after: (\d+\.\d+)', script_output).group(1)

  if "lora" in cpkt_path:
    is_lora = True
  else:
    is_lora = False
  steps = cpkt_path.split("sdxl_lightning_")[1].split("step")[0]

  benchmark_result_text = f"| {model_name} | {cpkt_path} | {steps} | {is_lora} | {height}x{width} | {iterations_per_second} | {inference_time} | {cuda_mem_after} | {host_mem_after} |\n"
  return benchmark_result_text

def benchmark_sd_model(model_name, model_path, repo_path, cpkt_path, resolutions, warmups, compiler):
  benchmark_result_text = "| Model | CPKT | Step | Is Lora | HxW | it/s | E2E Time (s) | CUDA Mem after (GiB) | Host Mem after (GiB) |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
  for resolution in resolutions.split(","):
    height, width = resolution.split("x")
    benchmark_result_text += benchmark_sd_model_with_one_resolution(model_name, model_path, repo_path, cpkt_path, warmups, compiler, height, width)
  return benchmark_result_text

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Benchmark script")
  parser.add_argument("-m", "--model_dir", help="the directory of the models, if not set, use HF models")
  parser.add_argument("-w", "--warmups", type=int, default=3, help="the number of warmups, default is 3")
  parser.add_argument("-c", "--compiler", default="oneflow", help="the compiler, default is oneflow")
  parser.add_argument("-o", "--output_file", default="/dev/stdout", help="the output file, default is /dev/stdout")
  args = parser.parse_args()

  MODEL_DIR = args.model_dir
  WARMUPS = args.warmups
  COMPILER = args.compiler
  OUTPUT_FILE = args.output_file

  SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

  SDXL_MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
  SDXL_LIGHT_PATH = "ByteDance/SDXL-Lightning"

  if MODEL_DIR is None:
    print("model_dir unspecified, use HF models")
  else:
    print("model_dir specified, use local models")
    MODEL_DIR = os.path.realpath(MODEL_DIR)
    if os.path.isdir(os.path.join(MODEL_DIR, "stable-diffusion-xl-base-1.0")):
      SDXL_MODEL_PATH = os.path.join(MODEL_DIR, "stable-diffusion-xl-base-1.0")
    if os.path.isdir(os.path.join(MODEL_DIR, "SDXL-Lightning")):
      SDXL_LIGHT_PATH = os.path.join(MODEL_DIR, "SDXL-Lightning")

  BENCHMARK_RESULT_TEXT = ""

  BENCHMARK_RESULT_TEXT += benchmark_sd_model("sdxl_light", SDXL_MODEL_PATH, SDXL_LIGHT_PATH, "sdxl_lightning_2step_unet.safetensors", "1024x1024", WARMUPS, COMPILER)
  BENCHMARK_RESULT_TEXT += benchmark_sd_model("sdxl_light", SDXL_MODEL_PATH, SDXL_LIGHT_PATH, "sdxl_lightning_4step_unet.safetensors", "1024x1024", WARMUPS, COMPILER)
  BENCHMARK_RESULT_TEXT += benchmark_sd_model("sdxl_light", SDXL_MODEL_PATH, SDXL_LIGHT_PATH, "sdxl_lightning_8step_unet.safetensors", "1024x1024", WARMUPS, COMPILER)

  BENCHMARK_RESULT_TEXT += benchmark_sd_model("sdxl_light", SDXL_MODEL_PATH, SDXL_LIGHT_PATH, "sdxl_lightning_2step_lora.safetensors", "1024x1024", WARMUPS, COMPILER)
  BENCHMARK_RESULT_TEXT += benchmark_sd_model("sdxl_light", SDXL_MODEL_PATH, SDXL_LIGHT_PATH, "sdxl_lightning_4step_lora.safetensors", "1024x1024", WARMUPS, COMPILER)
  BENCHMARK_RESULT_TEXT += benchmark_sd_model("sdxl_light", SDXL_MODEL_PATH, SDXL_LIGHT_PATH, "sdxl_lightning_8step_lora.safetensors", "1024x1024", WARMUPS, COMPILER)

  with open(OUTPUT_FILE, "w") as f:
    f.write(BENCHMARK_RESULT_TEXT)
