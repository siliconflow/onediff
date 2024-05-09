from onediff_benchmark import OneDiffBenchmark

if __name__ == "__main__":
    local_model_dir = "/data/home/wangerlie/onediff/benchmarks/models"
    out_dir = "/data/home/wangerlie/onediff/benchmarks"

    bench = OneDiffBenchmark(
        model_type="StableDiffusion",
        model_dir=local_model_dir,
        model_name="stable-diffusion-2-1",
        height=768,
        width=512,
    )
    bench.save_results(out_dir)

    bench = OneDiffBenchmark(
        model_type="StableDiffusion",
        model_dir=local_model_dir,
        model_name="stable-diffusion-v1-5",
        height=768,
        width=512,
    )
    bench.save_results(out_dir)

    bench = OneDiffBenchmark(
        model_type="StableDiffusionXL",
        model_dir=local_model_dir,
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
        height=512,
        width=512,
    )
    bench.save_results(out_dir)

    bench = OneDiffBenchmark(
        model_type="SDXL-Lightning",
        model_dir=local_model_dir,
        model_name="stable-diffusion-xl-base-1.0",
        height=512,
        width=512,
    )
    bench.save_results(out_dir)

    bench = OneDiffBenchmark(
        model_type="SVD",
        model_dir=local_model_dir,
        model_name="stabilityai/stable-video-diffusion-img2vid-xt",
        height=512,
        width=512,
    )
    bench.save_results(out_dir)
