from importlib_metadata import version

from onediff.utils import logger


def get_support_message():
    message = f""" OneDiff Enterprise Edition features can't be used, please refer to here for help: https://github.com/siliconflow/onediff?tab=readme-ov-file#community--support """
    return message


def is_quantization_enabled():
    import oneflow

    if version("oneflow") < "0.9.1":
        logger.warning(
            f"onediff_comfy_nodes requires oneflow>=0.9.1 to run, {get_support_message()}"
        )
        return False
    try:
        import onediff_quant
    except ImportError as e:
        logger.warning(
            f"Failed to import onediff_quant, Error message: {e}, {get_support_message()}"
        )
        return False
    return hasattr(oneflow._C, "dynamic_quantization")


def is_community_version():
    import oneflow.sysconfig

    return not oneflow.sysconfig.with_enterprise()
