import os
from sd_benchmark import StableDiffusionBenchmark
from sdxl_light_benchmark import SDXLLightBenchmark
from svd_benchmark import SVDBenchmark


MODEL_TYPE = ["StableDiffusion", "StableDiffusionXL", "SDXL-Lightning", "SVD"]


class OneDiffBenchmark:
    def __init__(
        self,
        model_type="StableDiffusion",
        model_dir=None,
        model_name=None,
        compiler="oneflow",
        height=512,
        width=512,
        variant="fp16",
        device="cuda",
        *args,
        **kwargs,
    ):
        self.model_type = model_type
        self.model_dir = model_dir
        self.model_name = model_name
        self.compiler = compiler
        self.height = height
        self.width = width
        self.variant = variant
        self.device = device

        self.benchmark = self._create_benchmark(
            model_type,
            model_dir,
            model_name,
            compiler,
            variant,
            height,
            width,
            device,
            *args,
            **kwargs,
        )

    def _create_benchmark(
        self,
        model_type,
        model_dir,
        model_name,
        compiler,
        variant,
        height,
        width,
        device,
        *args,
        **kwargs,
    ):
        if model_type in ["StableDiffusion", "StableDiffusionXL"]:
            return StableDiffusionBenchmark(
                model_dir=model_dir,
                model_name=model_name,
                compiler=compiler,
                variant=variant,
                height=height,
                width=width,
                device=device,
                *args,
                **kwargs,
            )
        elif model_type == "SDXL-Lightning":
            return SDXLLightBenchmark(
                model_dir=model_dir,
                model_name=model_name,
                compiler=compiler,
                variant=variant,
                height=height,
                width=width,
                *args,
                **kwargs,
            )
        elif model_type == "SVD":
            return SVDBenchmark(
                model_dir=model_dir,
                model_name=model_name,
                compiler=compiler,
                variant=variant,
                height=height,
                width=width,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Expected one of: StableDiffusion, StableDiffusionXL, SDXL-Lightning, SVD."
            )

    def get_results(self):
        print("Start Loading Pipeline from Diffusers.")
        self.benchmark.load_pipeline_from_diffusers()
        print("Start Compiling Pipeline.")
        self.benchmark.compile_pipeline()
        print("Start Benchmark")
        self.benchmark.benchmark_model()
        return self.benchmark.results

    def save_results(self, out_dir):
        results = self.get_results()
        info = f"{self.model_type}+{self.model_name} | {self.height}x{self.width} | {results['iter_per_sec']} | {results['inference_time']} | {results['cuda_mem_after_used']} | {results['host_mem_after_used']} |\n"
        # if the file 'results.md' doesn't exist, make the file and write the text head
        if not os.path.exists(os.path.join(out_dir, "results.md")):
            with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
                f.write(
                    "| Model | HxW | it/s | E2E Time (s) | CUDA Mem after (GiB) | Host Mem after (GiB) |\n| --- | --- | --- | --- | --- | --- |\n"
                )
        else:
            with open(os.path.join(out_dir, "results.md"), "a", encoding="utf-8") as f:
                f.write(info)


if __name__ == "__main__":
    x = OneDiffBenchmark(
        model_dir="/data/home/wangerlie/onediff/benchmarks/models",
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
    )
    results = x.get_results()
    print(results)
