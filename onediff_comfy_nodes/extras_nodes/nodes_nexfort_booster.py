import collections

from ..modules.nexfort.booster_basic import BasicNexFortBoosterExecutor


# https://github.com/siliconflow/nexfort?tab=readme-ov-file#suggested-combinations-of-compiler-modes
compiler_modes = collections.OrderedDict(
    {
        "max-optimize:max-autotune:low-precision": "This will deliver a good performance and adapt quickly to shape changes.",
        "max-optimize:max-autotune:low-precision:freezing:benchmark": "",
        "jit:disable-runtime-fusion:low-precision": "This compiles super quickly, but the performance might not be optimized very noticeably.",
        "jit:benchmark:low-precision:freezing:cudagraphs": "This compiles the model very quickly, but the performance might be not as good as `TorchInductor` optimized models.",
        "max-autotune:benchmark:low-precision:cudagraphs": "This is the most suggested combination of compiler modes. It will deliver a good balance between performance and compilation time.",
        "max-optimize:max-autotune:benchmark:low-precision:freezing:cudagraphs": "This is the most aggressive combination of compiler modes. It will deliver the best performance but might slow down the compilation significantly.",
    }
)


class OneDiffNexfortBooster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fullgraph": ([False, True],),
                "dynamic": ([True, False],),
                "mode": ([mode for mode in compiler_modes.keys()],),
                "docs_link": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[Note]: \nInstall-nexfort \nhttps://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort#install-nexfort",
                    },
                ),
            }
        }

    CATEGORY = "OneDiff/Booster"
    RETURN_TYPES = ("TorchCompileBooster",)
    FUNCTION = "apply"

    def apply(
        self,
        fullgraph=False,
        dynamic=None,
        mode="max-autotune:cudagraphs",
        docs_link=None,
    ):
        return (
            BasicNexFortBoosterExecutor(
                fullgraph=fullgraph, mode=f"{mode}:cache-all", dynamic=dynamic
            ),
        )


NODE_CLASS_MAPPINGS = {
    "OneDiffNexfortBooster": OneDiffNexfortBooster,
}

NODE_DISPLAY_NAME_MAPPINGS = {"OneDiffNexfortBooster": "Nexfort Booster - OneDiff"}
