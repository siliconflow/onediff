set -eux
rm -f unet_compiled
python3 test_load_sdxl.py  --save $@
python3 test_load_sdxl.py  --load $@
