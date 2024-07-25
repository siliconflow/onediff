import hashlib

from oneflow.framework.args_tree import ArgsTree


def extract_node_name(v):
    return type(v).__name__


def generate_input_structure_key(args_tree: ArgsTree):
    out_str = "_".join(
        (extract_node_name(node) for node in args_tree.iter_nodes() if node is not None)
    )
    return hashlib.sha256(out_str.encode("utf-8")).hexdigest()[:6]


def generate_model_structure_key(deployable_module):
    model = deployable_module._deployable_module_model.oneflow_module
    model_hash = hashlib.sha256(f"{model}".encode("utf-8")).hexdigest()
    return model_hash[:8]
