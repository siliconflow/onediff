import os
import time
import click
import oneflow as flow
from onediff import OneFlowStableDiffusionPipeline

prompt = "a photo of an astronaut riding a horse on mars"


def run_uncompile_graph(dim, test_round):
    print("Running uncompile pipe...")
    pipe = OneFlowStableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        revision="fp16",
        torch_dtype=flow.float16,
    )
    pipe = pipe.to("cuda")
    cur_generator = flow.Generator("cuda").manual_seed(1024)
    start = time.time()
    for _ in range(test_round):
        images = pipe(
            prompt,
            height=dim,
            width=dim,
            compile_unet=False,
            compile_vae=False,
            generator=cur_generator,
            num_inference_steps=50,
            output_type="np",
        ).images
    print(
        f"Average Time Cost for {test_round} times: {(time.time() - start) / test_round}s"
    )
    print(
        f"Average Throughput {test_round} times: {test_round * 50 / (time.time() - start):.2f}it/s"
    )
    print(f"Cuda Mem Use: {flow._oneflow_internal.GetCPUMemoryUsed()}MB")


def run_save_graph(pipe_file_path, graph_save_path, dim):
    os.makedirs(pipe_file_path, exist_ok=True)
    os.makedirs(graph_save_path, exist_ok=True)
    print("Running grave save...")
    pipe = OneFlowStableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        revision="fp16",
        torch_dtype=flow.float16,
    )
    pipe = pipe.to("cuda")
    pipe.set_graph_compile_cache_size(9)
    pipe.enable_graph_share_mem()
    cur_generator = flow.Generator("cuda").manual_seed(1024)
    start = time.time()
    images = pipe(
        prompt,
        height=dim,
        width=dim,
        generator=cur_generator,
        num_inference_steps=50,
        output_type="np",
    ).images
    pipe.save_pretrained(pipe_file_path)
    pipe.save_graph(graph_save_path)
    print(f"Save Time Cost: {time.time() - start:.2f}s")
    print(f"Cuda Mem Use: {flow._oneflow_internal.GetCPUMemoryUsed()}MB")


def run_load_graph(pipe_file_path, graph_save_path, dim, test_round):
    print("Running grave load...")
    pipe = OneFlowStableDiffusionPipeline.from_pretrained(
        pipe_file_path,
        revision="fp16",
        torch_dtype=flow.float16,
    )

    pipe = pipe.to("cuda")
    pipe.set_graph_compile_cache_size(9)

    pipe.load_graph(
        graph_save_path,
        warmup_with_run=True,
    )
    cur_generator = flow.Generator("cuda").manual_seed(1024)
    start = time.time()
    for _ in range(test_round):
        images = pipe(
            prompt,
            height=dim,
            width=dim,
            generator=cur_generator,
            num_inference_steps=50,
            output_type="np",
        ).images
    print(
        f"Average Throughput {test_round} times: {test_round * 50 / (time.time() - start):.2f}it/s"
    )
    print(f"Cuda Mem Use: {flow._oneflow_internal.GetCPUMemoryUsed()}MB")
    print(
        f"Average Time Cost for {test_round} times: {(time.time() - start) / test_round}s"
    )


@click.command()
@click.option("--pipe_path", default="./pipe", help="pipe file path")
@click.option("--graph_path", default="./graph", help="graph file path")
@click.option("--compile_graph", default=False, type=bool, help="compile graph or not")
@click.option("--dim", default=512, help="image dim")
@click.option("--test_num", default=20)
def run_test(pipe_path, graph_path, compile_graph, dim, test_num):
    if compile_graph:
        run_save_graph(pipe_path, graph_path, dim)
        flow.framework.session_context.TryCloseDefaultSession()
        time.sleep(5)
        flow.framework.session_context.NewDefaultSession(
            flow._oneflow_global_unique_env
        )
        run_load_graph(pipe_path, graph_path, dim, test_num)
    else:
        run_uncompile_graph(dim, test_num)


if __name__ == "__main__":
    run_test()
