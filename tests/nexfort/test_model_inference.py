import copy
import time
import unittest
from functools import partial

import torch

import onediff.infer_compiler as infer_compiler


class SubModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return self.lin(x)


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(1000, 100)
        self.sub_module = SubModule()

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear_layer(x))
        x = self.sub_module(x)
        return x


def foo(x):
    return torch.sin(x) + torch.cos(x)


class TestModelInference(unittest.TestCase):
    def setUp(self) -> None:
        self.compile_options = {
            "mode": "max-optimize:max-autotune:freezing:benchmark:cudagraphs",
            "dynamic": True,
            "fullgraph": True,
        }
        self.compile_fn = partial(
            infer_compiler.compile, backend="nexfort", options=self.compile_options
        )
        # self.compile_fn = torch.compile

    def measure_inference(self, model, warmup=3, num_runs=30, in_args=[], in_kwargs={}):
        # Warmup phase
        for _ in range(warmup):
            model(*in_args, **in_kwargs)

        # Timing
        total_time = 0.0
        for _ in range(num_runs):
            start_time = time.time()
            model(*in_args, **in_kwargs)
            total_time += time.time() - start_time
        result = model(*in_args, **in_kwargs)
        return result, total_time / num_runs

    def generate_models_and_inputs(self):
        model = MyModule().cuda().half()
        input_args = [torch.randn(10000, 1000).cuda().half()]
        compiled_model = self.compile_fn(model)
        yield model, compiled_model, input_args, {}

        model = MyModule().cuda().half()
        input_args = [torch.randn(10000, 1000).cuda().half()]
        compiled_model = copy.deepcopy(model)
        compiled_model.sub_module = self.compile_fn(compiled_model.sub_module)
        yield model, compiled_model, input_args, {}

    @torch.inference_mode()
    def test_model_inference(self):
        for (
            model,
            compiled_model,
            in_args,
            in_kwargs,
        ) in self.generate_models_and_inputs():

            original_result, original_time = self.measure_inference(
                model, in_args=in_args, in_kwargs=in_kwargs
            )
            print(f"Original model time: {original_time:.6f} seconds")

            compiled_result, compiled_time = self.measure_inference(
                compiled_model, in_args=in_args, in_kwargs=in_kwargs
            )
            print(f"Compiled model time: {compiled_time:.6f} seconds")

            self.assertTrue(
                torch.allclose(original_result, compiled_result, atol=1e-2, rtol=1e-3)
            )
            self.assertIsInstance(compiled_model, MyModule)
            self.assertEqual(
                set(model.state_dict().keys()), set(compiled_model.state_dict().keys())
            )

    @torch.inference_mode()
    def test_func_inference(self):
        compiled_foo = self.compile_fn(foo)
        input_args = [torch.randn(10, 100).cuda().half()]

        original_result, original_time = self.measure_inference(foo, in_args=input_args)

        compiled_result, compiled_time = self.measure_inference(
            compiled_foo, in_args=input_args
        )

        self.assertTrue(
            torch.allclose(original_result, compiled_result, atol=1e-3, rtol=1e-3)
        )


if __name__ == "__main__":
    unittest.main()
