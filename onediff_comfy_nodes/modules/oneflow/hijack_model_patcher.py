from comfy.model_patcher import ModelPatcher
from ..sd_hijack_utils import Hijacker
from .utils.booster_utils import is_using_oneflow_backend

def clone_oneflow(org_fn, self):
    n = ModelPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device, weight_inplace_update=self.weight_inplace_update)
    n.patches = {}
    for k in self.patches:
        n.patches[k] = self.patches[k][:]
    n.patches_uuid = self.patches_uuid

    n.object_patches = self.object_patches.copy()
    # TODO Security
    # n.model_options = copy.deepcopy(self.model_options)
    n.model_options = self.model_options
    
    n.model_keys = self.model_keys
    n.backup = self.backup
    n.object_patches_backup = self.object_patches_backup
    for attr_key in ["deep_cache_unet", "fast_deep_cache_unet"]:
        attr_value =  getattr(self, attr_key, None)
        if attr_value:
            setattr(n, attr_key,attr_value)
    return n

def cond_func(org_fn, self):
    return is_using_oneflow_backend(self)
    
model_patch_hijacker = Hijacker()

model_patch_hijacker.register(ModelPatcher.clone, clone_oneflow, cond_func)

