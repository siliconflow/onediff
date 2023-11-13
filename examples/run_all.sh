set -ex
python3 examples/text_to_image_sdxl.py --base /share_nfs/hf_models/stable-diffusion-xl-base-1.0 --compile
python3 benchmarks/stable_diffusion_2_unet.py --model_id=/share_nfs/hf_models/stable-diffusion-2-1
bash examples/unet_save_and_load.sh --model_id=/share_
nfs/hf_models/stable-diffusion-2-1
python3 examples/text_to_image.py --model_id=/share_nfs/hf_models/models--runwayml--stable-diffusion-v1-5
python3 -m onediff.demo
python3 examples/image_to_image.py --model_id=/share_nfs/hf_models/stable-diffusion-2-1
