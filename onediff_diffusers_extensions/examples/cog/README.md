RUN:

python3 onediff_diffusers_extensions/examples/cog/text_to_image_cog.py --model /data0/hf_models/CogVideoX-2b --compiler nexfort --compiler-config '{"mode": "max-optimize:max-autotune:max-autotune", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": false, "triton.fuse_attention_allow_fp16_reduction": false}}'

python3 onediff_diffusers_extensions/examples/cog/text_to_image_cog.py --model /data0/hf_models/CogVideoX-2b --compiler torch
