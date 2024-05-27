import os


def setup_onediff_quant():
    os.environ.setdefault("ONEFLOW_MLIR_GROUP_MATMUL_QUANT", "1")
    os.environ.setdefault("ONEFLOW_ATTENTION_ALLOW_QUANTIZATION", "1")
    os.environ.setdefault("ONEFLOW_KERNEL_GLU_QUANT_ENABLE_DUAL_GEMM_IMPL", "1")

    import onediff_quant

    onediff_quant.enable_load_quantized_model()


def load_calibration_and_quantize_pipeline(calibration_path, pipe):
    from onediff_quant.quantization import CalibrationStorage
    from onediff_quant.utils import replace_sub_module_with_quantizable_module

    store = CalibrationStorage()
    calibrate_info = store.load_from_file(file_path=calibration_path)

    for sub_module_name, sub_calibrate_info in calibrate_info.items():
        replace_sub_module_with_quantizable_module(
            pipe.unet,
            sub_module_name,
            sub_calibrate_info,
            fake_quant=False,
            static=False,
            nbits=8,
            convert_quant_module_fn=lambda x: x,
            original_module_name=None,
        )
