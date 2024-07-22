import os
import re
import time

from onediff.infer_compiler import DeployableModule

from .model_patcher import (
    OneFlowDeepCacheSpeedUpModelPatcher,
    OneFlowSpeedUpModelPatcher,
)
from .onediff_load_utils import onediff_load_quant_checkpoint_advanced
from .onediff_quant_utils import (
    quantize_and_save_model,
    replace_module_with_quantizable_module,
)

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "graphs"
)


def load_graph(deploy_module, graph_filename, device, subfolder):
    graph_filename = os.path.join(OUTPUT_FOLDER, subfolder, graph_filename)
    if isinstance(deploy_module, DeployableModule):
        deploy_module = deploy_module.to(device)
        start_time = time.time()
        if os.path.isfile(graph_filename):
            if deploy_module.get_graph().is_compiled:
                return

            print(f"Loading existed graph from {graph_filename} ...")
            deploy_module.load_graph(graph_filename)
            print(
                f"Graph has been loaded. {time.time() - start_time:.1f} seconds elapsed"
            )
        else:
            print(f"Invalid file to load: {graph_filename}")
    else:
        raise NotImplementedError(f"Unsupported: {type(deploy_module)}")


def save_graph(deploy_module, prefix: str, device: str, subfolder: str):
    str_cls_type = str(type(deploy_module._deployable_module_model._torch_module))
    match = re.search(r"<class '(.+?)'>", str_cls_type)
    module_class_name = match.group(1)

    graph_filename = os.path.join(
        OUTPUT_FOLDER,
        subfolder,
        f"{prefix}-{device}-{module_class_name}.graph",
    )

    if isinstance(deploy_module, DeployableModule):
        deploy_module = deploy_module.to(device)

        start_time = time.time()
        if not os.path.exists(graph_filename):
            if not hasattr(deploy_module.get_graph(), "inputs_original"):
                print(
                    f"Warning: The Graph loaded from the disk doesn't support saving, nor is it necessary to save. Skip saving."
                )
                return
            print(f"Graph saving...")
            deploy_module.save_graph(graph_filename)
            print(
                f"Graph saved to {graph_filename}. {time.time() - start_time:.1f} seconds elapsed"
            )
        else:
            print(
                f"File existed, please change the prefix if you want to save it: {graph_filename}"
            )
    else:
        raise NotImplementedError(f"Unsupported: {type(deploy_module)}")
