from ..modules.torch_compile.booster_basic import (  # type: ignore
    TorchCompileBoosterExecutor,
)


class OneDiffTorchCompileBooster:
    @classmethod
    def INPUT_TYPES(s):
        # [docs]https://pytorch.org/docs/stable/_modules/torch.html#compile
        return {
            "required": {
                "fullgraph": ([False, True],),
                "dynamic": ([None, True, False],),
                "backend": (["inductor"],),
                "mode": (
                    [
                        "default",
                        "reduce-overhead",
                        "max-autotune",
                        "max-autotune-no-cudagraphs",
                    ],
                ),
                "disable": ([False, True],),
                "docs_link": (
                    "STRING",
                    {
                        "multiline": True,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "[docs]: https://pytorch.org/docs/stable/_modules/torch.html#compile",
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
        backend="inductor",
        mode="default",
        options=None,
        disable=False,
        docs_link=None,
    ):
        return (
            TorchCompileBoosterExecutor(
                fullgraph=fullgraph,
                dynamic=dynamic,
                backend=backend,
                mode=mode,
                options=options,
                disable=disable,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "OneDiffTorchCompileBooster": OneDiffTorchCompileBooster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OneDiffTorchCompileBooster": "Torch Compile Booster - OneDiff"
}
