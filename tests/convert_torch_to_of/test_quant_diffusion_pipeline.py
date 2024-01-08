import pytest
import torch
from onediff.infer_compiler import oneflow_compile
from mock_onediff_quant import QuantDiffusionPipeline


@pytest.mark.parametrize("model", ["/ssd/home/hanbinbin/sdxl-1.0-base-int8"])
@pytest.mark.parametrize("fake_quant", [False])
@pytest.mark.parametrize("static", [False])
@pytest.mark.parametrize("bits", [8])
@pytest.mark.parametrize("graph", [True])
@pytest.mark.parametrize(
    "prompt",
    ['"street style, detailed, raw photo, woman, face, shot on CineStill 800T"'],
)
def test_quant_diffusion_pipeline(model, fake_quant, static, bits, graph, prompt):
    pipe = QuantDiffusionPipeline.from_pretrained(
        model, fake_quant, static, bits, graph
    )
    pipe.to("cuda")
    pipe.unet = oneflow_compile(pipe.unet)
    image = pipe(prompt, height=512, width=512).images[0]
    image.save("test_quant_diffusion_pipeline.png")
