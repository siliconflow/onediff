
    

## Experimental Data
<details close>
<summary> stable-diffusion-xl-base-1.0 step-20_image-1024x1024 </summary>

[stable-diffusion-xl-base-1.0 step-20_image-1024x1024](https://ccssu.github.io/git_pages.io/htmls/sdxl_step-20_image-1024x1024_calibrate_info_unet.html) 
<!-- import torch 
from diffusers import AutoPipelineForText2Image
floatting_model_path = "/share_nfs/hf_models/stable-diffusion-xl-base-1.0"
from onediff.quantization import QuantPipeline
pipe = QuantPipeline.from_pretrained(
    AutoPipelineForText2Image, floatting_model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")
prompt = "street style, detailed, raw photo, woman, face, shot on CineStill 800T"

pipe_kwargs = dict(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=30,
)
pipe.quantize(**pipe_kwargs,
    conv_compute_density_threshold=900,
    linear_compute_density_threshold=300,
    conv_ssim_threshold=0.985,
    linear_ssim_threshold=0.991,
    save_as_float=False,
    cache_dir="run_sdxl_quant_") -->
</details>