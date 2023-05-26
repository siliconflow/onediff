import time
import click
import torch
from diffusers import DiffusionPipeline

prompt = "a photo of an astronaut riding a horse on mars"


def run_diffusers(dim, compile, test_round):
    device = torch.cuda.current_device()
    print("Running diffusers pipe...")
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    if compile:
        print("Using torch compile")
        pipe.unet = torch.compile(pipe.unet)

    images = pipe(
        prompt,
        height=dim,
        width=dim,
        num_inference_steps=50,
    ).images
    start = time.time()
    for i in range(test_round):
        images = pipe(
            prompt,
            height=dim,
            width=dim,
            num_inference_steps=50,
        ).images
        memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Cuda Mem Use: {memory_allocated}MB")

    print(
        f"Average Time Cost for {test_round} times: {(time.time() - start) / test_round:.3f}s"
    )
    print(
        f"Average Throughput {test_round} times: {test_round * 50 / (time.time() - start):.2f}it/s"
    )


@click.command()
@click.option("--dim", default=1024, help="image dim")
@click.option("--compile", default=True, help="compile option of torch")
@click.option("--test_num", default=20)
def run_test(dim, compile, test_num):
    run_diffusers(dim, compile, test_num)


if __name__ == "__main__":
    run_test()
