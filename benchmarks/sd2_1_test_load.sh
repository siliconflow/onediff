set -eux
rm -f unet_graphs
python3 ../examples/save_and_load_graph.py --model_id /share_nfs/hf_models/stable-diffusion-2-1/ --save
python3 ../examples/save_and_load_graph.py --model_id /share_nfs/hf_models/stable-diffusion-2-1/ --load
