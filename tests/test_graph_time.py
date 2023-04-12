import os
from onediff import OneFlowStableDiffusionPipeline
import time
import oneflow as flow

flow.mock_torch.enable()
pipe_file_path = "./pipe"
graph_save_path = "./graph"
DIM = 896
prompt = "a photo of an astronaut riding a horse on mars"
cur_generator = flow.Generator("cuda").manual_seed(1024)


def run_save_graph(pipe_file_path, graph_save_path):
    os.makedirs(pipe_file_path, exist_ok=True)
    os.makedirs(graph_save_path, exist_ok=True)
    print("Running grave save...")
    pipe = OneFlowStableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        revision="fp16",
        torch_dtype=flow.float16,
    )
    pipe = pipe.to("cuda")
    pipe.set_graph_compile_cache_size(9)
    pipe.enable_graph_share_mem()

    start = time.time()
    images = pipe(
        prompt,
        height=DIM,
        width=DIM,
        compile_unet=True,
        compile_vae=True,
        generator=cur_generator,
        output_type="np",
    ).images
    pipe.save_pretrained(pipe_file_path)
    pipe.save_graph(graph_save_path)
    print(f"Time cost: {time.time() - start:.2f}s")


def run_load_graph(pipe_file_path, graph_save_path):
    print("Running grave load...")
    pipe = OneFlowStableDiffusionPipeline.from_pretrained(
        pipe_file_path,
        revision="fp16",
        torch_dtype=flow.float16,
    )

    pipe = pipe.to("cuda")
    pipe.set_graph_compile_cache_size(9)

    pipe.load_graph(
        graph_save_path, compile_unet=True, compile_vae=True, time_flag=True
    )

    start = time.time()
    images = pipe(
        prompt,
        height=DIM,
        width=DIM,
        compile_unet=True,
        compile_vae=True,
        generator=cur_generator,
        output_type="np",
    ).images

    print(f"Time cost (first round): {time.time() - start:.2f}s")
    start = time.time()
    images = pipe(
        prompt,
        height=DIM,
        width=DIM,
        compile_unet=True,
        compile_vae=True,
        generator=cur_generator,
        output_type="np",
    ).images

    print(f"Time cost (second round): {time.time() - start:.2f}s")


if __name__ == "__main__":
    run_save_graph(pipe_file_path, graph_save_path)
    flow.framework.session_context.TryCloseDefaultSession()
    time.sleep(5)
    flow.framework.session_context.NewDefaultSession(flow._oneflow_global_unique_env)
    run_load_graph(pipe_file_path, graph_save_path)
