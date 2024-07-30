import copy
import time
import unittest
from functools import partial

import onediff.infer_compiler as infer_compiler

import torch
from onediff.utils.import_utils import is_nexfort_available, is_oneflow_available


class SubModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 10)

    def forward(self, x):
        return self.linear(x)


class MainModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1000, 100)
        self.sub_module = SubModule()

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear(x))
        x = self.sub_module(x)
        return x


def compute(x):
    return torch.sin(x) + torch.cos(x)


class TestModelInference(unittest.TestCase):
    def setUp(self) -> None:
        self.compilation_functions = []

        if is_oneflow_available():
            oneflow_compile_fn = partial(infer_compiler.compile, backend="oneflow")
            self.compilation_functions.append(oneflow_compile_fn)

        if is_nexfort_available():
            nexfort_compile_options = {
                "mode": "max-optimize:max-autotune:freezing:benchmark:cudagraphs",
                "dynamic": True,
                "fullgraph": True,
            }
            nexfort_compile_fn = partial(
                infer_compiler.compile,
                backend="nexfort",
                options=nexfort_compile_options,
            )
            self.compilation_functions.append(nexfort_compile_fn)

        assert len(self.compilation_functions) > 0

    def measure_inference_time(
        self, model, warmup=3, num_runs=30, input_args=[], input_kwargs={}
    ):
        for _ in range(warmup):
            model(*input_args, **input_kwargs)

        total_time = 0.0
        for _ in range(num_runs):
            start_time = time.time()
            model(*input_args, **input_kwargs)
            total_time += time.time() - start_time

        average_time = total_time / num_runs
        result = model(*input_args, **input_kwargs)
        return result, average_time

    def generate_models_and_inputs(self):
        for compile_fn in self.compilation_functions:
            model = MainModule().cuda().half()
            inputs = [torch.randn(10000, 1000).cuda().half()]
            compiled_model = compile_fn(model)
            yield model, compiled_model, inputs, {}

            model = MainModule().cuda().half()
            inputs = [torch.randn(10000, 1000).cuda().half()]
            compiled_model_sub = copy.deepcopy(model)
            compiled_model_sub.sub_module = compile_fn(compiled_model_sub.sub_module)
            yield model, compiled_model_sub, inputs, {}

            if compile_fn.keywords.get("backend") == "nexfort":
                inputs_compute = [torch.randn(10000, 1000).cuda().half()]
                compiled_compute_fn = compile_fn(compute)
                yield compute, compiled_compute_fn, inputs_compute, {}

    @torch.inference_mode()
    def test_inference_results(self):
        for (
            model,
            compiled_model,
            input_args,
            input_kwargs,
        ) in self.generate_models_and_inputs():
            original_result, _ = self.measure_inference_time(
                model, input_args=input_args, input_kwargs=input_kwargs
            )
            compiled_result, _ = self.measure_inference_time(
                compiled_model, input_args=input_args, input_kwargs=input_kwargs
            )

            self.assertTrue(
                torch.allclose(original_result, compiled_result, atol=1e-2, rtol=1e-3)
            )

            if isinstance(model, torch.nn.Module):
                self.assertIsInstance(compiled_model, MainModule)
                self.assertEqual(
                    set(model.state_dict().keys()),
                    set(compiled_model.state_dict().keys()),
                )


if __name__ == "__main__":
    unittest.main()
