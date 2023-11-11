set -eux
rm -f unet_compiled
python3 test_load_speed.py  --save $@
python3 test_load_speed.py  --load $@
