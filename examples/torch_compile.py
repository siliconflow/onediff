from diffusers import StableDiffusionPipeline

import torch
from unet_torch_interplay import MockCtx
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
)


def add_op_from_torch(node):
    getattr(oneflow, node["functioname"])(node.args)

global num_graphs
num_graphs = 0
torch._dynamo.config.verbose=True

def to_of_transform(
    gm: torch.fx.GraphModule
) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if not node.is_impure():
            raise "Graph impure, can't convert to oneflow"
        if node.op == "call_function":
            if node.target in mapping_dict:
                node.target = mapping_dict[node.target]
            else:
                raise NotImplementedError
        elif node.op == "call_method":
            if hasattr(torch.Tensor, node.target):
                pass
            else:
                raise NotImplementedError

    gm.graph.lint()
    with MockCtx():
        gm.recompile()
    return gm

def torchbackend(gm, example_inputs):
    import oneflow as flow
    import torch_fx_to_oneflow
    torch_fx_to_oneflow.to_of_transform(gm)
    # TODO: when initialzing oneflow variables, find them in the state dict and reuse them using dlpack
    gm.print_readable()
    # g = flow.NNGraph()
    # for node in gm.graph.nodes:
        # g.add_op_from_torch(node)
        # Checks if we're calling a function (i.e:
        # torch.add)
        # if node.op == "call_function":
        # print(node.op, node)
    # g.compile()
    global num_graphs
    num_graphs += 1
    return gm.forward

# print(pipe.unet.state_dict().keys())
pipe.unet = torch.compile(pipe.unet, fullgraph=True, mode="reduce-overhead", backend=torchbackend)
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True, backend=my_custom_backend)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with torch.autocast("cuda"):
    images = pipe(prompt).images
    print(f"{num_graphs=}")
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
