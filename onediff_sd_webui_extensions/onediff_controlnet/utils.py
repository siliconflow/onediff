def check_if_controlnet_ext_loaded() -> bool:
    from modules import extensions

    return "sd-webui-controlnet" in extensions.loaded_extensions


def get_controlnet_script(p):
    for script in p.scripts.scripts:
        if script.__module__ == "controlnet.py":
            return script
    return None


def check_if_controlnet_enabled(p):
    controlnet_script_class = get_controlnet_script(p)
    return (
        controlnet_script_class is not None
        and len(controlnet_script_class.get_enabled_units(p)) != 0
    )
