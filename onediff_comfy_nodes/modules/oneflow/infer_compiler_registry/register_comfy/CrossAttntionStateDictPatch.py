from typing import Dict


def process_state_dict_before_saving(state_dict: Dict, graph=None):
    assert graph is not None

    try:
        for graph_key, oneflow_graph in state_dict.items():
            attn2_patches = (
                oneflow_graph.get("inputs_original", [])[1]
                .get("transformer_options", {})
                .get("patches_replace", {})
                .get("attn2")
            )
            if attn2_patches:

                del oneflow_graph["inputs_original"][1]["transformer_options"][
                    "patches_replace"
                ]["attn2"]

    except Exception as e:
        print(f"Warning: Failed to process state dict before saving: {e}")

    return state_dict


def attn2_patch_sum(input_kwargs) -> int:
    attn2 = (
        input_kwargs.get("transformer_options", {})
        .get("patches_replace", {})
        .get("attn2", {})
    )
    return len(attn2)
