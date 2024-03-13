"""
The code is mainly referenced from https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_stable_diffusion.py
"""

import argparse


# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0", "time_embedding.linear_1"),
    ("time_embed.2", "time_embedding.linear_2"),
    ("input_blocks.0.0", "conv_in"),
    ("out.0", "conv_norm_out"),
    ("out.2", "conv_out"),
    ("input_blocks.0.0", "conv_in"),
    ("time_embed.0", "time_embedding.linear_1"),
    ("time_embed.2", "time_embedding.linear_2"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(3):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))



def convert_unet_calibrate_dict(state_dict) -> str:
    mapping = {k: k for k in state_dict}
    for sd_name, hf_name in unet_conversion_map:
        if hf_name in mapping:
            mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            sd_part, hf_part = sd_part.removesuffix("."), hf_part.removesuffix(".")
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: state_dict[k] for k, v in mapping.items()}
    return new_state_dict

def convert_calibrate_info(calibration_path, dst_path):
    calibrate_info = {}
    with open(calibration_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            items = line.split(" ")
            calibrate_info[items[0]] = " ".join(items[1: ])
    dst_info = convert_unet_calibrate_dict(calibrate_info)
    with open(dst_path, "w") as f:
        for name, info in dst_info.items():
            f.write(f"{name} {info}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="./calibrate_info.txt", type=str, required=True, help="Path to the calibrate info to convert.")
    parser.add_argument("--dst", default="./sd_calibrate_info.txt", type=str, required=True, help="Path to save the sd format calibrate info.")
    args = parser.parse_args()
    convert_calibrate_info(args.src, args.dst)
