from ..modules.nexfort.booster_basic import BasicNexFortBoosterExecutor


class OneDiffNexfortBooster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fullgraph": ([False, True],),
                "dynamic": ([None, True, False],),
                "mode": (
                    [
                        "max-optimize:max-autotune:freezing:benchmark:cudagraphs",
                        "max-autotune:cudagraphs",
                        "cudagraphs:benchmark:low-precision:freezing:jit"
                        "max-optimize:max-autotune:cudagraphs:benchmark:low-precision:freezing",
                    ],
                ),
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
                fullgraph=fullgraph, mode=mode, dynamic=dynamic
            ),
        )


NODE_CLASS_MAPPINGS = {
    "OneDiffNexfortBooster": OneDiffNexfortBooster,
}

NODE_DISPLAY_NAME_MAPPINGS = {"OneDiffNexfortBooster": "Nexfort Booster - OneDiff"}
