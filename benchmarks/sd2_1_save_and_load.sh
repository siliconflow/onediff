set -eux
rm -f unet_graphs
python3 ./sd2_1_save_and_load.py --save $@
python3 ./sd2_1_save_and_load.py --load $@
