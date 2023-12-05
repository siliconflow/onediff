import oneflow
from importlib_metadata import version


def get_support_message():
    recipient_email = "caishenghang@oneflow.org"

    message = f"""\033[91m Advanced features cannot be used !!! 
        If you need unrestricted multiple resolution, quant support or any other more advanced
        features, please send an email to {recipient_email} and tell us about 
        your **use case, deployment scale and requirements**.
        """
    return message


def is_quantization_enabled():
    if version("oneflow") < "0.9.1":
        RuntimeError(
            "onediff_comfy_nodes requires oneflow>=0.9.1 to run.", get_support_message()
        )
        return False
    try:
        import diffusers_quant
    except ImportError:
        return False
    return hasattr(oneflow._C, "dynamic_quantization")


def is_community_version(stop_if_not=False):
    is_community = not is_quantization_enabled()
    if is_community:
        message = get_support_message()
        if stop_if_not:
            input(message + "\nPress any key to continue...")
        else:
            print(message)
    return is_community
