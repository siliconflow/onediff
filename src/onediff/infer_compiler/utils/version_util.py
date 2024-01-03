from .log_utils import logger


def get_support_message():
    recipient_email = "business@siliconflow.com"

    message = f"""\033[91m Advanced features cannot be used !!! \033[0m
If you need unrestricted multiple resolution, quantization support or any other more advanced features, please send an email to \033[91m{recipient_email}\033[0m and tell us about your use case, deployment scale and requirements.
        """
    return message


def is_oneflow_pro() -> bool:
    try:
        import oneflow.sysconfig as sysconfig

        return sysconfig.with_enterprise()
    except Exception as e:
        logger.warning(f"Unable to determine if OneFlow is Pro or not. {e}")
        logger.warning(get_support_message())
        return False


def is_community_version():
    return not is_oneflow_pro()


def is_quantization_enabled() -> bool:
    if not is_oneflow_pro():
        return False
    try:
        import diffusers_quant

        return True
    except Exception as e:
        logger.warning(f"Quantization is not enabled. {e}")
        logger.warning(get_support_message())
        return False
