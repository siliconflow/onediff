[1mdiff --git a/.github/workflows/examples.yml b/.github/workflows/examples.yml[m
[1mindex a70b10b7..f4ca3390 100644[m
[1m--- a/.github/workflows/examples.yml[m
[1m+++ b/.github/workflows/examples.yml[m
[36m@@ -298,12 +298,14 @@[m [mjobs:[m
             }[m
           }[m
 [m
[31m-          run_comfy_test "workflows/sdxl-unet-speedup-graph-saver.json" 200[m
[31m-          run_comfy_test "workflows/sdxl-control-lora-speedup.json" 200[m
[31m-          run_comfy_test "/share_nfs/hf_models/comfyui_resources/workflows/ipadapter_advanced.json" 200[m
[31m-          run_comfy_test "/share_nfs/hf_models/comfyui_resources/workflows/deep-cache.json" 600[m
[31m-          run_comfy_test "/share_nfs/hf_models/comfyui_resources/workflows/deep-cache-with-lora.json" 800[m
[32m+[m[32m          # run_comfy_test "workflows/sdxl-unet-speedup-graph-saver.json" 200[m
[32m+[m[32m          # run_comfy_test "workflows/sdxl-control-lora-speedup.json" 200[m
[32m+[m[32m          # run_comfy_test "/share_nfs/hf_models/comfyui_resources/workflows/ipadapter_advanced.json" 200[m
[32m+[m[32m          # run_comfy_test "/share_nfs/hf_models/comfyui_resources/workflows/deep-cache.json" 600[m
[32m+[m[32m          # run_comfy_test "/share_nfs/hf_models/comfyui_resources/workflows/deep-cache-with-lora.json" 800[m
           # run_comfy_test "workflows/text-to-video-speedup.json" 5000[m
[32m+[m[32m          docker exec -w /src/onediff/onediff_comfy_nodes/benchmarks ${{ env.CONTAINER_NAME }} bash scripts/run_oneflow_case.sh[m
[32m+[m
 [m
       - name: Show ComfyUI Log[m
         if: matrix.test-suite == 'comfy'[m
[1mdiff --git a/onediff_comfy_nodes/benchmarks/resources/example_workflow_api.json b/onediff_comfy_nodes/benchmarks/resources/example_workflow_api.json[m
[1mindex 670f67eb..a487d5cb 100644[m
[1m--- a/onediff_comfy_nodes/benchmarks/resources/example_workflow_api.json[m
[1m+++ b/onediff_comfy_nodes/benchmarks/resources/example_workflow_api.json[m
[36m@@ -22,7 +22,7 @@[m
             ],[m
             "sampler_name": "euler",[m
             "scheduler": "normal",[m
[31m-            "seed": 8566257,[m
[32m+[m[32m            "seed": 1,[m
             "steps": 20[m
         }[m
     },[m
[1mdiff --git a/onediff_comfy_nodes/benchmarks/scripts/install_env.sh b/onediff_comfy_nodes/benchmarks/scripts/install_env.sh[m
[1mindex 43d46f71..ac86760e 100644[m
[1m--- a/onediff_comfy_nodes/benchmarks/scripts/install_env.sh[m
[1m+++ b/onediff_comfy_nodes/benchmarks/scripts/install_env.sh[m
[36m@@ -1,3 +1,4 @@[m
 pip install websocket-client==1.8.0[m
 pip install pynvml==11.5.0[m
[32m+[m[32mpip install numpy==1.26.4[m
 pip install scikit-image[m
\ No newline at end of file[m
[1mdiff --git a/onediff_comfy_nodes/benchmarks/scripts/run_text_to_image.sh b/onediff_comfy_nodes/benchmarks/scripts/run_text_to_image.sh[m
[1mindex ea2e36aa..6c484003 100644[m
[1m--- a/onediff_comfy_nodes/benchmarks/scripts/run_text_to_image.sh[m
[1m+++ b/onediff_comfy_nodes/benchmarks/scripts/run_text_to_image.sh[m
[36m@@ -7,6 +7,9 @@[m
 [m
 # Run the Python script[m
 # python3 scripts/text_to_image.py -w resources/example_workflow_api.json --comfy-pid 1438896[m
[31m-comfy_pid=2222[m
[32m+[m[32mcomfy_pid=3688922[m
 # python3 scripts/text_to_image.py -w  resources/baseline/sd3_baseline.json  --comfy-pid $comfy_pid  --output-images[m
[31m-python3 scripts/text_to_image.py -w  resources/nexfort/sd3_unet_vae_speedup.json --comfy-pid $comfy_pid # --output-images[m
[32m+[m[32m# python3 scripts/text_to_image.py -w  resources/nexfort/sd3_unet_vae_speedup.json --comfy-pid $comfy_pid # --output-images[m
[32m+[m
[32m+[m
[32m+[m[32mpython3 scripts/text_to_image.py -w  resources/oneflow/sdxl-control-lora-speedup.json --comfy-pid $comfy_pid # --output-images[m
[1mdiff --git a/onediff_comfy_nodes/benchmarks/src/input_registration.py b/onediff_comfy_nodes/benchmarks/src/input_registration.py[m
[1mindex fd92c944..58622a35 100644[m
[1m--- a/onediff_comfy_nodes/benchmarks/src/input_registration.py[m
[1m+++ b/onediff_comfy_nodes/benchmarks/src/input_registration.py[m
[36m@@ -1,4 +1,5 @@[m
 import json[m
[32m+[m[32mimport os[m
 from typing import NamedTuple[m
 from core import create_constructor_registry, ComfyGraph, logger[m
 [m
[36m@@ -8,6 +9,10 @@[m [mdef read_prompts(file_path="resources/prompts.txt"):[m
         lines = fp.readlines()[m
     return [line.strip() for line in lines if line.strip()][m
 [m
[32m+[m[32mdef get_all_images(directory, image_extensions = set(['.jpg', '.jpeg', '.png', '.gif', '.bmp'])):[m
[32m+[m[32m    all_files = [os.path.join(directory, f) for f in os.listdir(directory)][m
[32m+[m[32m    image_files = [f for f in all_files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in image_extensions][m
[32m+[m[32m    return image_files[m
 [m
 # Create register and get functions[m
 register_constructor, get_input_constructor = create_constructor_registry()[m
[36m@@ -26,14 +31,11 @@[m [mdef _(workflow_api_file_path):[m
             graph.set_image_size(height=height, width=width)[m
             yield graph[m
 [m
[31m-[m
 SD3_WORKFLOWS = [[m
     "resources/baseline/sd3_baseline.json",[m
     "resources/nexfort/sd3_unet_speedup.json",[m
     "resources/nexfort/sd3_unet_vae_speedup.json",[m
 ][m
[31m-[m
[31m-[m
 @register_constructor(SD3_WORKFLOWS)[m
 def _(workflow_api_file_path):[m
     with open(workflow_api_file_path, "r") as fp:[m
[36m@@ -48,3 +50,34 @@[m [mdef _(workflow_api_file_path):[m
                 logger.info(f"ComfyGraph: {height=} {width=}")[m
                 graph.set_image_size(height=height, width=width)[m
                 yield graph[m
[32m+[m
[32m+[m
[32m+[m[32m@register_constructor("resources/oneflow/sdxl-control-lora-speedup.json")[m
[32m+[m[32mdef _(workflow_api_file_path):[m
[32m+[m[32m    with open(workflow_api_file_path, "r") as fp:[m
[32m+[m[32m        workflow = json.load(fp)[m
[32m+[m
[32m+[m[32m    graph = ComfyGraph(graph=workflow, sampler_nodes=["1"])[m
[32m+[m[32m    yield graph[m
[32m+[m
[32m+[m
[32m+[m
[32m+[m
[32m+[m[32m@register_constructor(["resources/baseline/ComfyUI_IPAdapter_plus/ipadapter_advanced.json",[m[41m [m
[32m+[m[32m                       "resources/oneflow/ComfyUI_IPAdapter_plus/ipadapter_advanced.json"])[m
[32m+[m[32mdef _(workflow_api_file_path):[m
[32m+[m[32m    with open(workflow_api_file_path, "r") as fp:[m
[32m+[m[32m        workflow = json.load(fp)[m
[32m+[m
[32m+[m[32m    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])[m
[32m+[m
[32m+[m[32m    for image in get_all_images("/share_nfs/hf_models/comfyui_resources/input/faces"):[m
[32m+[m[32m        graph.graph["12"]["inputs"]["image"] = image[m
[32m+[m[32m        print(f'{image}')[m
[32m+[m[32m        for height in [768, 512]:[m
[32m+[m[32m            for width in [768, 512]:[m
[32m+[m[32m                logger.info(f"ComfyGraph: {height=} {width=}")[m
[32m+[m[32m                graph.set_image_size(height=height, width=width)[m
[32m+[m[32m                yield graph[m
[32m+[m
[32m+[m[32m    yield graph[m
\ No newline at end of file[m
[1mdiff --git a/tests/comfyui/extra_model_paths.yaml b/tests/comfyui/extra_model_paths.yaml[m
[1mindex 007a395e..c47714ea 100644[m
[1m--- a/tests/comfyui/extra_model_paths.yaml[m
[1m+++ b/tests/comfyui/extra_model_paths.yaml[m
[36m@@ -8,7 +8,7 @@[m [mcomfyui:[m
     base_path: /app/ComfyUI[m
     # checkpoints: /home/fengwen/workspace/test_checkpoints[m
     checkpoints: /share_nfs/hf_models/comfyui_resources/checkpoints[m
[31m-    clip: models/clip/[m
[32m+[m[32m    clip: /share_nfs/hf_models/comfyui_resources//models/clip/[m
     clip_vision: /share_nfs/hf_models/comfyui_resources/clip_vision[m
     configs: models/configs/[m
     controlnet: models/controlnet/[m
