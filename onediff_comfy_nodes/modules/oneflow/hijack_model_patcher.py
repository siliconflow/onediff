import copy
from comfy.model_patcher import ModelPatcher
from ..sd_hijack_utils import Hijacker
from .utils.booster_utils import is_using_oneflow_backend
from register_comfy.CrossAttentionPatch import CrossAttentionPatch
from .patch_management import PatchType, create_patch_executor

def extract_and_clone_non_cross_attention(original_dict):
    # Initialize a new dictionary for storing extracted values
    new_dict = {}

    # Iterate over the original dictionary
    for key, value in original_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recursively process it
            new_value = extract_and_clone_non_cross_attention(value)
        elif isinstance(value, CrossAttentionPatch):
            new_value = value
        else:
            # Otherwise, perform a deep copy of the value
            new_value = copy.deepcopy(value)
        
        # Update the new dictionary with the processed value
        new_dict[key] = new_value
    return new_dict

def clone_oneflow(org_fn, self):    
    n = ModelPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device, weight_inplace_update=self.weight_inplace_update)
    n.patches = {}
    for k in self.patches:
        n.patches[k] = self.patches[k][:]
    n.patches_uuid = self.patches_uuid

    n.object_patches = self.object_patches.copy()
    diff_model = self.model.diffusion_model
    cc_patch_executor = create_patch_executor(PatchType.C_C_Patch)
    if cc_patch_executor.check_patch(diff_model):
        n.model_options = extract_and_clone_non_cross_attention(self.model_options)
    else:
        n.model_options = copy.deepcopy(self.model_options)
    
    create_patch_executor(PatchType.CrossAttentionUpdatePatch).copy_to(self, n)
    
    n.model_keys = self.model_keys
    n.backup = self.backup
    n.object_patches_backup = self.object_patches_backup
    
    dc_patch_executor = create_patch_executor(PatchType.DCUNetExecutorPatch)
    if dc_patch_executor.check_patch(self):
        dc_patch_executor.copy_to(self, n)
    return n

def cond_func(org_fn, self):
    return is_using_oneflow_backend(self)
    
model_patch_hijacker = Hijacker()

model_patch_hijacker.register(ModelPatcher.clone, clone_oneflow, cond_func)

