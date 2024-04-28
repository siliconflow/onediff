#!/usr/bin/env python3

import argparse
import os
import re
import subprocess

def benchmark_sd_model_with_one_resolution(model_name, model_path, warmups, compiler, height, width):
  print(f"Run {model_path} {height}x{width}...")
  if "deepcache" in model_name:
    script_output = subprocess.check_output([
      "python3",
      os.path.join(SCRIPT_DIR, "text_to_image.py"),
      "--model", model_path,
      "--variant", "fp16",
      "--warmups", str(warmups),
      "--compiler", compiler,
      "--height", str(height),
      "--width", str(width),
      "--deepcache"
    ]).decode("utf-8")
  else:
    script_output = subprocess.check_output([
      "python3",
      os.path.join(SCRIPT_DIR, "text_to_image.py"),
      "--model", model_path,
      "--variant", "fp16",
      "--warmups", str(warmups),
      "--compiler", compiler,
      "--height", str(height),
      "--width", str(width)
    ]).decode("utf-8")


  inference_time = re.search(r"Inference time: (\d+\.\d+)", script_output).group(1)
  iterations_per_second = re.search(r"Iterations per second: (\d+\.\d+)", script_output).group(1)
  cuda_mem_after = re.search(r"CUDA Mem after: (\d+\.\d+)", script_output).group(1)
  host_mem_after = re.search(r"Host Mem after: (\d+\.\d+)", script_output).group(1)

  benchmark_result_text.append(f"| {model_name} | {height}x{width} | {iterations_per_second} | {inference_time} | {cuda_mem_after} | {host_mem_after} |\n")

def benchmark_sd_model(model_name, model_path, resolutions, warmups, compiler):
  for resolution in resolutions:
    height, width = resolution.split("x")
    benchmark_sd_model_with_one_resolution(model_name, model_path, warmups, compiler, height, width)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run text to image benchmark")
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

  SD15_MODEL_PATH = "runwayml/stable-diffusion-v1-5"
  SD21_MODEL_PATH = "stabilityai/stable-diffusion-2-1"
  SDXL_MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"

  BENCHMARK_QUANT_MODEL = 0
  BENCHMARK_DEEP_CACHE_MODEL = 0

  if MODEL_DIR is None:
    print("model_dir unspecified, use HF models")
  else:
    print("model_dir specified, use local models")
    MODEL_DIR = os.path.realpath(MODEL_DIR)
    if os.path.isdir(os.path.join(MODEL_DIR, "stable-diffusion-v1-5")):
      SD15_MODEL_PATH = os.path.join(MODEL_DIR, "stable-diffusion-v1-5")
    if os.path.isdir(os.path.join(MODEL_DIR, "stable-diffusion-2-1")):
      SD21_MODEL_PATH = os.path.join(MODEL_DIR, "stable-diffusion-2-1")
    if os.path.isdir(os.path.join(MODEL_DIR, "stable-diffusion-xl-base-1.0")):
      SDXL_MODEL_PATH = os.path.join(MODEL_DIR, "stable-diffusion-xl-base-1.0")

    try:
      import onediff_quant
      print("enable quant model")
      BENCHMARK_QUANT_MODEL = 1
    except ImportError:
      print("disable quant model")

    # try:
    #     import onediffx
    #     print("enable deepcache model")
    #     BENCHMARK_DEEP_CACHE_MODEL = 1
    # except ImportError:
    #     print("disable deepcache model")

    SDXL_QUANT_MODEL_PATH = os.path.join(MODEL_DIR, "stable-diffusion-xl-base-1.0-int8")
    SDXL_DEEP_CACHE_QUANT_MODEL_PATH = os.path.join(MODEL_DIR, "stable-diffusion-xl-base-1.0-deepcache-int8")

  BENCHMARK_RESULT_TEXT = "| Model | HxW | it/s | E2E Time (s) | CUDA Mem after (GiB) | Host Mem after (GiB) |\n| --- | --- | --- | --- | --- | --- |\n"
  benchmark_result_text = []

  benchmark_sd_model("sd15", SD15_MODEL_PATH, ["1024x1024", "720x1280", "768x768", "512x512"], WARMUPS, COMPILER)
  benchmark_sd_model("sd21", SD21_MODEL_PATH, ["1024x1024", "720x1280", "768x768", "512x512"], WARMUPS, COMPILER)
  benchmark_sd_model("sdxl", SDXL_MODEL_PATH, ["1024x1024", "720x1280", "768x768", "512x512"], WARMUPS, COMPILER)

  if BENCHMARK_QUANT_MODEL != 0 and COMPILER == "oneflow":
    benchmark_sd_model("sdxl_quant", SDXL_QUANT_MODEL_PATH, ["1024x1024", "720x1280", "768x768", "512x512"], WARMUPS, COMPILER)

  if BENCHMARK_QUANT_MODEL != 0 and BENCHMARK_DEEP_CACHE_MODEL != 0 and COMPILER == "oneflow":
    benchmark_sd_model("sdxl_deepcache_quant", SDXL_DEEP_CACHE_QUANT_MODEL_PATH, ["1024x1024", "720x1280", "768x768", "512x512"], WARMUPS, COMPILER)

  with open(OUTPUT_FILE, "w") as f:
    f.write(BENCHMARK_RESULT_TEXT)
    f.writelines(benchmark_result_text)
