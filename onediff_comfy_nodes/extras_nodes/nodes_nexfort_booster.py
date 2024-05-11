from ..modules.nexfort.booster_basic import BasicNexFortBoosterExecutor

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class OneDiffNexfortBooster:
    
    @classmethod
    def INPUT_TYPES(s):
        return {}
    
    CATEGORY = "OneDiff/Booster"
    RETURN_TYPES = ("TorchCompileBooster",)
    FUNCTION = "apply"

    def apply(self, *args, **kwargs):
        return (BasicNexFortBoosterExecutor(),)