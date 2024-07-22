import comfy
import comfy.utils
import torch
import torchvision.transforms as T
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from register_comfy.CrossAttentionPatch import pulid_attention

from ..hijack_ipadapter_plus.set_model_patch_replace import apply_patch
from ..utils.booster_utils import is_using_oneflow_backend
from ._config import pulid_comfyui_hijacker, pulid_comfyui_pt

pulid_pkg = pulid_comfyui_pt.pulid
PulidModel = pulid_pkg.PulidModel
tensor_to_image = pulid_pkg.tensor_to_image
image_to_tensor = pulid_pkg.image_to_tensor
to_gray = pulid_pkg.to_gray


def set_model_patch_replace(model, patch_kwargs, key):
    apply_patch(
        org_fn=None,
        model=model,
        patch_kwargs=patch_kwargs,
        key=key,
        attention_func=pulid_attention,
    )


def apply_pulid(
    org_fn,
    self,
    model,
    pulid,
    eva_clip,
    face_analysis,
    image,
    weight,
    start_at,
    end_at,
    method=None,
    noise=0.0,
    fidelity=None,
    projection=None,
    attn_mask=None,
):
    work_model = model.clone()

    device = comfy.model_management.get_torch_device()
    dtype = comfy.model_management.unet_dtype()
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        dtype = (
            torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        )

    eva_clip.to(device, dtype=dtype)
    # pulid_model = PulidModel(pulid).to(device, dtype=dtype)
    if hasattr(self, "_pulid_model"):
        cached_pulid, pulid_model = self._pulid_model
        if cached_pulid == pulid:
            print(f"Model is cached.")
        else:
            print(f"Cached model not found for pulid: {pulid}.")
            pulid_model = PulidModel(pulid).to(device, dtype=dtype)
            self._pulid_model = (pulid, pulid_model)
    else:
        pulid_model = PulidModel(pulid).to(device, dtype=dtype)
        self._pulid_model = (pulid, pulid_model)
        print(f"Cached Model")

    if attn_mask is not None:
        if attn_mask.dim() > 3:
            attn_mask = attn_mask.squeeze(-1)
        elif attn_mask.dim() < 3:
            attn_mask = attn_mask.unsqueeze(0)
        attn_mask = attn_mask.to(device, dtype=dtype)

    if method == "fidelity" or projection == "ortho_v2":
        num_zero = 8
        ortho = False
        ortho_v2 = True
    elif method == "style" or projection == "ortho":
        num_zero = 16
        ortho = True
        ortho_v2 = False
    else:
        num_zero = 0
        ortho = False
        ortho_v2 = False

    if fidelity is not None:
        num_zero = fidelity

    # face_analysis.det_model.input_size = (640,640)
    image = tensor_to_image(image)

    if hasattr(self, "_face_helper_info"):
        face_helper = self._face_helper_info
    else:
        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            device=device,
        )

        face_helper.face_parse = None
        face_helper.face_parse = init_parsing_model(model_name="bisenet", device=device)
        self._face_helper_info = face_helper

    bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
    cond = []
    uncond = []

    for i in range(image.shape[0]):
        # get insightface embeddings
        iface_embeds = None
        for size in [(size, size) for size in range(640, 256, -64)]:
            face_analysis.det_model.input_size = size
            face = face_analysis.get(image[i])
            if face:
                face = sorted(
                    face,
                    key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                    reverse=True,
                )[-1]
                iface_embeds = (
                    torch.from_numpy(face.embedding)
                    .unsqueeze(0)
                    .to(device, dtype=dtype)
                )
                break
        else:
            raise Exception("insightface: No face detected.")

        # get eva_clip embeddings
        face_helper.clean_all()
        face_helper.read_image(image[i])
        face_helper.get_face_landmarks_5(only_center_face=True)
        face_helper.align_warp_face()

        if len(face_helper.cropped_faces) == 0:
            raise Exception("facexlib: No face detected.")

        face = face_helper.cropped_faces[0]
        face = image_to_tensor(face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        parsing_out = face_helper.face_parse(
            T.functional.normalize(face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(face)
        face_features_image = torch.where(bg, white_image, to_gray(face))
        face_features_image = T.functional.resize(
            face_features_image,
            eva_clip.image_size,
            T.InterpolationMode.BICUBIC
            if "cuda" in device.type
            else T.InterpolationMode.BILINEAR,
        ).to(device, dtype=dtype)
        face_features_image = T.functional.normalize(
            face_features_image, eva_clip.image_mean, eva_clip.image_std
        )

        id_cond_vit, id_vit_hidden = eva_clip(
            face_features_image,
            return_all_features=False,
            return_hidden=True,
            shuffle=False,
        )
        id_cond_vit = id_cond_vit.to(device, dtype=dtype)
        for idx in range(len(id_vit_hidden)):
            id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

        id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

        # combine embeddings
        id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)
        if noise == 0:
            id_uncond = torch.zeros_like(id_cond)
        else:
            id_uncond = torch.rand_like(id_cond) * noise
        id_vit_hidden_uncond = []
        for idx in range(len(id_vit_hidden)):
            if noise == 0:
                id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[idx]))
            else:
                id_vit_hidden_uncond.append(torch.rand_like(id_vit_hidden[idx]) * noise)

        cond.append(pulid_model.get_image_embeds(id_cond, id_vit_hidden))
        uncond.append(pulid_model.get_image_embeds(id_uncond, id_vit_hidden_uncond))

    # average embeddings
    cond = torch.cat(cond).to(device, dtype=dtype)
    uncond = torch.cat(uncond).to(device, dtype=dtype)
    if cond.shape[0] > 1:
        cond = torch.mean(cond, dim=0, keepdim=True)
        uncond = torch.mean(uncond, dim=0, keepdim=True)

    if num_zero > 0:
        if noise == 0:
            zero_tensor = torch.zeros(
                (cond.size(0), num_zero, cond.size(-1)), dtype=dtype, device=device
            )
        else:
            zero_tensor = (
                torch.rand(
                    (cond.size(0), num_zero, cond.size(-1)), dtype=dtype, device=device
                )
                * noise
            )
        cond = torch.cat([cond, zero_tensor], dim=1)
        uncond = torch.cat([uncond, zero_tensor], dim=1)

    sigma_start = work_model.get_model_object("model_sampling").percent_to_sigma(
        start_at
    )
    sigma_end = work_model.get_model_object("model_sampling").percent_to_sigma(end_at)

    patch_kwargs = {
        "pulid": pulid_model,
        "weight": weight,
        "cond": cond,
        "uncond": uncond,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
        "ortho": ortho,
        "ortho_v2": ortho_v2,
        "mask": attn_mask,
    }

    number = 0
    for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
        block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
        for index in block_indices:
            patch_kwargs["module_key"] = str(number * 2 + 1)
            set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
            number += 1
    for id in range(6):  # id of output_blocks that have cross attention
        block_indices = range(2) if id in [3, 4, 5] else range(10)  # transformer_depth
        for index in block_indices:
            patch_kwargs["module_key"] = str(number * 2 + 1)
            set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
            number += 1
    for index in range(10):
        patch_kwargs["module_key"] = str(number * 2 + 1)
        set_model_patch_replace(work_model, patch_kwargs, ("middle", 0, index))
        number += 1

    return (work_model,)


def apply_pulid_cond_func(org_fn, self, model, *args, **kwargs):
    return is_using_oneflow_backend(model)


pulid_comfyui_hijacker.register(
    pulid_comfyui_pt.pulid.ApplyPulid.apply_pulid, apply_pulid, apply_pulid_cond_func
)
