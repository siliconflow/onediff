fix: https://github.com/siliconflow/onediff/issues/795


https://github.com/cubiq/ComfyUI_InstantID/tree/main


mkdir -p ComfyUI/models/instantid

/home/fengwen/quant/datasets/InstantX/InstantID/ip-adapter.bin

/home/fengwen/quant/datasets/InstantX/InstantID/ControlNetModel/diffusion_pytorch_model.safetensors

wget -O  ComfyUI/models/v1-5-pruned-emaonly.ckpt  https://hf-mirror.com/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt


FAQ:
 Q: 1. Not sure if insightface is using CPU or GPU?
 A: https://github.com/deepinsight/insightface/issues/2394#issuecomment-1929310317