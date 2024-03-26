def unload_model_weights(sd_model=None, info=None):
    from modules import lowvram, devices
    from modules import shared
    m = sd_model or shared.sd_model
    if m.lowvram:
        lowvram.send_everything_to_cpu()
    else:
        m.to(devices.cpu)
    devices.torch_gc()
    return sd_model

def send_model_to_cpu(m):
    # do nothing
    pass

def hijack_function(module, name, new_name, new_value):
    # restore original function in case of reload
    unhijack_function(module=module, name=name, new_name=new_name)
    setattr(module, new_name, getattr(module, name))
    setattr(module, name, new_value)


def unhijack_function(module, name, new_name):
    if hasattr(module, new_name):
        setattr(module, name, getattr(module, new_name))
        delattr(module, new_name)

def do_hijack():
    from modules import sd_models, script_callbacks
    script_callbacks.on_script_unloaded(undo_hijack)
    hijack_function(
        module=sd_models,
        name='unload_model_weights',
        new_name='__onediff_original_unload_model_weights',
        new_value=unload_model_weights,
    )
    hijack_function(
        module=sd_models,
        name='send_model_to_cpu',
        new_name='__onediff_original_send_model_to_cpu',
        new_value=send_model_to_cpu,
    )

def undo_hijack():
    from modules import sd_models
    unhijack_function(
        module=sd_models,
        name='unload_model_weights',
        new_name='__onediff_original_unload_model_weights',
    )
    unhijack_function(
        module=sd_models,
        name='send_model_to_cpu',
        new_name='__onediff_original_send_model_to_cpu',
    )
