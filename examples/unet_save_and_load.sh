set -eu
python3 examples/unet_torch_interplay.py --save
python3 examples/unet_torch_interplay.py --load
