import modules.scripts as scripts
from modules import script_callbacks
import modules.shared as shared
from modules.processing import process_images
import math
import torch
import gradio as gr
import oneflow as flow
from einops import rearrange
from oneflow import nn, einsum
from sgm.modules.attention import default, CrossAttention
from sgm.modules.diffusionmodules.util import GroupNorm32
from omegaconf import OmegaConf, ListConfig
from onediff.infer_compiler.transform.builtin_transform import torch2oflow
from onediff.infer_compiler import oneflow_compile, register
from onediff.optimization.quant_optimizer import quantize_model


@torch2oflow.register
def _(mod, verbose=False) -> ListConfig:
    converted_list = [torch2oflow(item, verbose) for item in mod]
    return OmegaConf.create(converted_list)


"""oneflow_compiled UNetModel"""
_compiled = None
_first_quant = True


# https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_optimizations.py#L142
# https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_optimizations.py#L221
class CrossAttentionOflow(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        q_in = self.to_q(x)
        context = default(context, x)

        # context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
        context_k, context_v = context, context
        k_in = self.to_k(context_k)
        v_in = self.to_v(context_v)

        dtype = q_in.dtype
        # from modules import shared
        # if shared.opts.upcast_attn:
        #     q_in, k_in, v_in = q_in.float(), k_in.float(), v_in if v_in.device.type == 'mps' else v_in.float()

        # with devices.without_autocast(disable=not shared.opts.upcast_attn):
        k_in = k_in * self.scale

        del context, x

        q, k, v = (
            rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (q_in, k_in, v_in)
        )
        del q_in, k_in, v_in

        r1 = flow.zeros(
            q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype
        )

        # mem_free_total = get_available_vram()
        from modules import sd_hijack_optimizations

        mem_free_total = sd_hijack_optimizations.get_available_vram()

        gb = 1024**3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1

        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(
                f"Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). "
                f"Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free"
            )

        slice_size = q.shape[1] // steps
        for i in range(0, q.shape[1], slice_size):
            end = min(i + slice_size, q.shape[1])
            s1 = einsum("b i d, b j d -> b i j", q[:, i:end], k)

            # s2 = s1.softmax(dim=-1, dtype=q.dtype)
            s2 = s1.softmax(dim=-1)
            del s1

            r1[:, i:end] = einsum("b i j, b j d -> b i d", s2, v)
            del s2

        del q, k, v

        r1 = r1.to(dtype)

        r2 = rearrange(r1, "(b h) n d -> b n (h d)", h=h)
        del r1

        return self.to_out(r2)


# https://github.com/Stability-AI/generative-models/blob/e5963321482a091a78375f3aeb2c3867562c913f/sgm/modules/diffusionmodules/wrappers.py#L24
def forward_wrapper(self, x, t, c, **kwargs):
    x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
    with torch.autocast("cuda", enabled=False):
        with flow.autocast("cuda", enabled=False):
            return self.diffusion_model(
                x.half(),
                timesteps=t.half(),
                context=c.get("crossattn", None).half(),
                y=c.get("vector", None).half(),
                **kwargs,
            )


# https://github.com/Stability-AI/generative-models/blob/059d8e9cd9c55aea1ef2ece39abf605efb8b7cc9/sgm/modules/diffusionmodules/util.py#L274
class GroupNorm32Oflow(nn.GroupNorm):
    def forward(self, x):
        # return super().forward(x.float()).type(x.dtype)
        return super().forward(x).type(x.dtype)


# https://github.com/Stability-AI/generative-models/blob/e5963321482a091a78375f3aeb2c3867562c913f/sgm/modules/diffusionmodules/openaimodel.py#L983-L984
class TimeEmbedModule(nn.Module):
    def __init__(self, time_embed):
        super().__init__()
        self._time_embed_module = time_embed

    def forward(self, t_emb):
        return self._time_embed_module(t_emb.half())


torch2oflow_class_map = {
    CrossAttention: CrossAttentionOflow,
    GroupNorm32: GroupNorm32Oflow,
}
register(package_names=["sgm"], torch2oflow_class_map=torch2oflow_class_map)


def compile(sd_model, quantization=False):
    unet_model = sd_model.model.diffusion_model
    if quantization:
        # replace_module_with_quantizable_module
        quantize_model(unet_model, inplace=True)

    full_name = f"{unet_model.__module__}.{unet_model.__class__.__name__}"
    if full_name != "sgm.modules.diffusionmodules.openaimodel.UNetModel":
        return
    global _compiled
    _compiled = oneflow_compile(sd_model.model.diffusion_model, use_graph=True)
    time_embed_wrapper = TimeEmbedModule(
        _compiled._deployable_module_model.oneflow_module.time_embed
    )
    # https://github.com/Stability-AI/generative-models/blob/e5963321482a091a78375f3aeb2c3867562c913f/sgm/modules/diffusionmodules/openaimodel.py#L984
    setattr(
        _compiled._deployable_module_model.oneflow_module,
        "time_embed",
        time_embed_wrapper,
    )


class Script(scripts.Script):
    def title(self):
        return "onediff_diffusion_model"

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        return [gr.components.Checkbox(label="Model Quantization(int8) Speed Up")]

    def show(self, is_img2img):
        return not is_img2img

    def run(self, p, quantization):
        global _compiled, _first_quant

        if _compiled is None or _first_quant:
            _first_quant = False
            compile(shared.sd_model, quantization)
        # compile(shared.sd_model)
        original = shared.sd_model.model.diffusion_model
        from sgm.modules.diffusionmodules.wrappers import OpenAIWrapper

        orig_forward = OpenAIWrapper.forward
        if _compiled is not None:
            shared.sd_model.model.diffusion_model = _compiled
            setattr(OpenAIWrapper, "forward", forward_wrapper)
        proc = process_images(p)
        shared.sd_model.model.diffusion_model = original
        setattr(OpenAIWrapper, "forward", orig_forward)
        return proc


script_callbacks.on_model_loaded(compile)
