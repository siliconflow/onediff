### This script is used to test the benchmarking of StableDiffusion models.
### The benchmarking is done by running the model on a range of steps and measuring the inference time.
### https://github.com/siliconflow/sd-team/issues/401

import argparse

import oneflow
import onediff
from sd_benchmark import StableDiffusionBenchmark

## The end-to-end threshold for each model and size(1024x1024, 720x1280, 768x768, 512x512)
SD15_THRESHOLD = [3.4, 3, 2, 0.8]
SD21_THRESHOLD = [3.7, 3.4, 2, 0.8]
SDXL_THRESHOLD = [4.5, 4.2, 3, 1.5]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir", type=str, default=None, help="Path to the model directory"
)
parser.add_argument("--model_name", type=str, help="Name of the model to benchmark")
parser.add_argument(
    "--height", type=int, default=None, help="Height of the input image"
)
parser.add_argument("--width", type=int, default=None, help="Width of the input image")
parser.add_argument(
    "--e2e_threshold",
    type=float,
    default=2,
    help="Set end-to-end threshold for the model",
)
args = parser.parse_args()

print("===========info of oneflow and onediff============")
print("oneflow version:", oneflow.__version__)
print("onediff version:", onediff.__version__)
print("==================================================")
e2e_thresholds_list = []
if args.height is not None and args.width is not None:
    size = [(args.height, args.width)]
    e2e_thresholds_list.append(args.e2e_threshold)
else:
    size = [(1024, 1024), (720, 1280), (768, 768), (512, 512)]
    # TODO set the end-to-end threshold for each size
    if args.model_name == "stable-diffusion-v1-5":
        e2e_thresholds_list.extend(SD15_THRESHOLD)
    elif args.model_name == "stable-diffusion-2-1":
        e2e_thresholds_list.extend(SD21_THRESHOLD)
    elif args.model_name == "stable-diffusion-xl-base-1.0":
        e2e_thresholds_list.extend(SDXL_THRESHOLD)
    else:
        raise ValueError(f"Thresholds of model '{args.model_name}' have not been set.")
results = {}
results["model_name"] = args.model_name
for (h, w), e2e_threshold in zip(size, e2e_thresholds_list):
    print(f"Start benchmarking {args.model_name} model with size {h}x{w}")
    benchmark = StableDiffusionBenchmark(
        model_dir=args.model_dir,
        model_name=args.model_name,
        compiler="oneflow",
        height=h,
        width=w,
        deepcache=True,
    )
    benchmark.load_pipeline_from_diffusers()
    benchmark.compile_pipeline()
    benchmark.benchmark_model()
    results[f"image_size:{h}x{w}"] = benchmark.results
    inference_time = benchmark.results["inference_time"]
    if inference_time > e2e_threshold:
        raise ValueError(
            f"Model {args.model_name}: Inference time of size {h}x{w} is '{inference_time}' longer than the threshold '{e2e_threshold}'."
        )
    print(f"Finish benchmarking {args.model_name} model with size {h}x{w}")
    print("=======================================")
# print results, including inference time, iterations per second,
# CUDA memory usage, host memory usage, average throughput, and base time without base cost
print("Finish All Benchmarking")
print(results)
print("==================================================")
