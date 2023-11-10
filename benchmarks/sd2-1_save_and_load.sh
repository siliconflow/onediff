set -eux
rm -f unet_graphs
python3 ../examples/unet_torch_interplay.py --save --model_id /share_nfs/hf_models/stable-diffusion-2-1/
python3 ../examples/unet_torch_interplay.py --load --model_id /share_nfs/hf_models/stable-diffusion-2-1/
