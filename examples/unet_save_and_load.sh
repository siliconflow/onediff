set -eux
rm -f unet_graphs
python3 examples/unet_torch_interplay.py --save $@
python3 examples/unet_torch_interplay.py --load $@
