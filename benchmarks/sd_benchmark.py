import os
import torch
import oneflow as flow
from onediffx import compile_pipe
from benchmark_base import BaseBenchmark
from utils.sd_utils import *


class StableDiffusionBenchmark(BaseBenchmark):
    def __init__(
        self,
        model_dir=None,
        model_name="runwayml/stable-diffusion-v1-5",
        compiler="oneflow",
        out_dir=None,
        variant="fp16",
        custom_pipeline=None,
        scheduler="EulerAncestralDiscreteScheduler",
        lora=None,
        controlnet=None,
        torch_dtype=torch.float16,
        device="cuda",
        height=512,
        width=512,
        steps=30,
        batch=1,
        prompt="best quality, realistic, unreal engine, 4K, a beautiful",
        negative_prompt=None,
        seed=None,
        warmups=3,
        extra_call_kwargs=None,
        deepcache=False,
        cache_interval=3,
        cache_layer_id=0,
        cache_block_id=0,
        input_image=None,
        control_image=None,
        output_image=None,
        conmpiler_config=None,
        *args,
        **kwargs,
    ):
        self.model_dir = model_dir
        self.model_name = model_name
        self.compiler = compiler
        self.out_dir = out_dir
        self.variant = variant
        self.custom_pipeline = custom_pipeline
        self.scheduler = scheduler
        self.lora = lora
        self.controlnet = controlnet
        self.torch_dtype = torch_dtype
        self.height = height
        self.width = width
        self.steps = steps
        self.batch = batch
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.warmups = warmups
        self.extra_call_kwargs = extra_call_kwargs
        self.deepcache = deepcache
        self.cache_interval = cache_interval
        self.cache_layer_id = cache_layer_id
        self.cache_block_id = cache_block_id
        self.input_image = input_image
        self.control_image = control_image
        self.output_image = output_image

        self.device = get_device(device)
        from diffusers import AutoPipelineForText2Image as pipeline_cls

        if self.input_image is None and "deepcache" in self.model_name:
            from onediffx.deep_cache import StableDiffusionXLPipeline as pipeline_cls
        self.pipeline_cls = pipeline_cls
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.results = {}

    def load_pipeline_from_diffusers(self):
        if self.model_dir is not None:
            print("Use Local Model.")
            self.ckpt_path = os.path.join(
                self.model_dir, self.model_name.split("/")[-1]
            )
            print(f"Loading model from {self.ckpt_path}")
            if os.path.exists(self.ckpt_path):
                self.pipe = load_sd_pipe(
                    self.pipeline_cls,
                    self.ckpt_path,
                    self.torch_dtype,
                    self.variant,
                    self.custom_pipeline,
                    self.scheduler,
                    self.lora,
                    self.controlnet,
                    self.device,
                )
            else:
                raise ValueError(f"Model path {self.ckpt_path} does not exist")
        else:
            print("Use HF Model.")
            self.pipe = load_sd_pipe(
                self.pipeline_cls,
                self.model_name,
                self.torch_dtype,
                self.variant,
                self.custom_pipeline,
                self.scheduler,
                self.lora,
                self.controlnet,
                self.device,
            )

    def compile_pipeline(self):
        if self.compiler is None:
            pass
        elif self.compiler == "oneflow":
            self.pipe = compile_pipe(self.pipe)
            print("Compile pipeline with OneFlow")
        # compile with nexfort
        elif self.compiler == "nexfort":
            if self.kwargs.get("compiler_config") is not None:
                # config with dict
                options = json.loads(self.kwargs["compiler_config"])
            else:
                # config with string
                options = '{"mode": "max-optimize:max-autotune:freezing:benchmark:cudagraphs", "memory_format": "channels_last"}'
            pipe = compile_pipe(
                pipe, backend="nexfort", options=options, fuse_qkv_projections=True
            )
        elif self.compiler in ("compile", "compile-max-autotune"):
            mode = "max-autotune" if self.compiler == "compile-max-autotune" else None
            self.pipe.unet = torch.compile(self.pipe.unet, mode=mode)
            if hasattr(self.pipe, "controlnet"):
                self.pipe.controlnet = torch.compile(self.pipe.controlnet, mode=mode)
            self.pipe.vae = torch.compile(self.pipe.vae, mode=mode)
        else:
            raise ValueError(f"Unknown compiler: {self.compiler}")

    def benchmark_model(self):

        self.kwarg_inputs = get_kwarg_inputs(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            height=self.height,
            width=self.width,
            steps=self.steps,
            batch=self.batch,
            seed=self.seed,
            extra_call_kwargs=self.extra_call_kwargs,
            deepcache=self.deepcache,
            cache_interval=self.cache_interval,
            cache_layer_id=self.cache_layer_id,
            cache_block_id=self.cache_block_id,
            input_image=self.input_image,
            control_image=self.control_image,
        )
        if self.warmups > 0:
            print("Begin warmup")
            for _ in range(self.warmups):
                self.pipe(**self.kwarg_inputs)
            print("End warmup")
        # Let"s see it!
        # Note: Progress bar might work incorrectly due to the async nature of CUDA.
        iter_profiler = IterationProfiler()
        if "callback_on_step_end" in inspect.signature(self.pipe).parameters:
            self.kwarg_inputs["callback_on_step_end"] = (
                iter_profiler.callback_on_step_end
            )
        elif "callback" in inspect.signature(self.pipe).parameters:
            self.kwarg_inputs["callback"] = iter_profiler.callback_on_step_end
        begin = time.time()
        output_images = self.pipe(**self.kwarg_inputs).images
        end = time.time()

        print("=======================================")
        print(f"Inference time: {end - begin:.3f}s")
        self.results["inference_time"] = end - begin
        iter_per_sec = iter_profiler.get_iter_per_sec()
        if iter_per_sec is not None:
            print(f"Iterations per second: {iter_per_sec:.3f}")
            self.results["iter_per_sec"] = iter_per_sec
        cuda_mem_after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        host_mem_after_used = flow._oneflow_internal.GetCPUMemoryUsed()
        print(f"CUDA Mem after: {cuda_mem_after_used / 1024:.3f}GiB")
        print(f"Host Mem after: {host_mem_after_used / 1024:.3f}GiB")
        self.results["cuda_mem_after_used"] = cuda_mem_after_used / 1024
        self.results["host_mem_after_used"] = host_mem_after_used / 1024
        print("=======================================")

    def throughput_benchmark(self, steps_range=range(1, 100, 1)):
        (
            self.results["average_throughput"],
            self.results["base_time_without_base_cost"],
        ) = generate_data_and_fit_model(
            self.prompt, self.height, self.width, steps_range, self.pipe
        )
