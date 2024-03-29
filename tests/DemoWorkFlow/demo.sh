set -e 

cd src

# 指定 缓存图文件位置， 默认 `path/to/ComfyUI/input/graphs`:
# export COMFYUI_ONEDIFF_SAVE_GRAPH_DIR=path/to/cache_dir

export COMFYUI_ROOT=/home/worksplace/ComfyUI # path/to/ComfyUI
# torch
python sdxl_demo.py 
# onediff
python sdxl_demo.py --use_onediff